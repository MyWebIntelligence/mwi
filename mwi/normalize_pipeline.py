"""Retroactive URL normalization pipeline.

The on-line equivalent of `mwi.url_normalizer` is applied at insertion time
(`core.add_expression`). This module handles the **retroactive** case:
walking an existing Land and bringing every Expression up to the current
canonicalization rules.

Expressions are planned **by canonical-URL group**: every variant whose
normalization yields the same target URL belongs to one group.

  * **Rename** — the group's single member (or its promoted winner, see
    below) gets UPDATE'd in place, `original_url` populated. Optionally
    clear `http_status` / `fetched_at` so the crawl picks them up again.
  * **Merge** — when a row already holds the canonical URL, every variant
    is merged into it: remap every `ExpressionLink` touching the duplicate
    to the canonical, drop self-loops and pre-existing duplicates,
    backfill the canonical's empty content fields from the duplicate
    (never overwriting), then delete the redundant Expression. CASCADE
    removes the duplicate's Media, Paragraph, and TaggedContent rows.
  * **Promotion** — when several variants converge on a canonical URL that
    exists on NO row (e.g. http/www variants with force_https/strip_www
    newly enabled), the richest variant (html > readable > relevance >
    fetched_at > smallest id) is renamed to the canonical URL and the
    others are merged into it. Without this, all variants would be
    renamed to the same URL without ever being merged (Expression.url is
    a non-unique index).

Each pair processed in its own `DB.atomic()`, so the script is safely
interruptible and re-runnable (converges). At production scale (tens of
thousands of pairs) expect one transaction per pair. Chains (Wayback of
Wayback that span multiple Expressions) are resolved before any
modification — structurally impossible since `normalize_url` is
idempotent, kept as defense in depth.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from . import core, model
from .url_normalizer import normalize_url


def _promotion_key(row: Dict) -> Tuple:
    """Sort key electing the richest variant of a collision group.

    html beats readable beats relevance beats fetched_at; ties break on the
    smallest id (via -id under max()). What the winner lacks, the merge
    backfill recovers from the losers.
    """
    return (bool(row['has_html']), bool(row['has_readable']),
            row['relevance'] or 0, row['fetched_at'] is not None,
            -row['id'])


def _collect_pairs(land: model.Land) -> Tuple[List[Tuple[int, str, str, bool]],
                                              List[Tuple[int, str, int, str]],
                                              Dict[str, int]]:
    """Plan the normalization by canonical-URL group. Returns three values:

    - `to_rename`: tuples (expr_id, old_url, new_url, promoted). The row is
      UPDATE'd in place; `promoted` marks the winner of a collision group.
    - `to_merge`: tuples (dup_id, dup_url, canon_id, canon_url). The
      duplicate is merged into the canonical row and deleted.
    - `stats`: {'collision_groups': N} — groups with several variants.

    Only planning columns are selected (ids, urls, presence flags) — the
    html/readable payloads of the whole land are never materialized in RAM.
    Rows already holding their canonical URL are stable (normalize_url is
    idempotent) and never appear in a group. Limitation: two rows sharing
    the SAME already-canonical URL are left untouched.
    """
    Expr = model.Expression
    rows = list(Expr.select(
        Expr.id, Expr.url,
        Expr.html.is_null(False).alias('has_html'),
        (Expr.readable.is_null(False) & (Expr.readable != ''))
        .alias('has_readable'),
        Expr.relevance, Expr.fetched_at,
    ).where(Expr.land == land).dicts())

    url_to_row: Dict[str, Dict] = {r['url']: r for r in rows}

    groups: Dict[str, List[Dict]] = {}
    for r in rows:
        new_url = normalize_url(r['url'])
        if new_url != r['url']:
            groups.setdefault(new_url, []).append(r)

    to_rename: List[Tuple[int, str, str, bool]] = []
    to_merge: List[Tuple[int, str, int, str]] = []
    collision_groups = 0

    for new_url, members in groups.items():
        holder = url_to_row.get(new_url)
        if holder is not None:
            # a row already holds the canonical URL: merge every variant
            for m_row in members:
                to_merge.append((m_row['id'], m_row['url'],
                                 holder['id'], new_url))
        else:
            # no holder: promote the richest variant, merge the others
            winner = max(members, key=_promotion_key)
            to_rename.append((winner['id'], winner['url'], new_url,
                              len(members) > 1))
            for m_row in members:
                if m_row['id'] != winner['id']:
                    to_merge.append((m_row['id'], m_row['url'],
                                     winner['id'], new_url))
        if len(members) > 1:
            collision_groups += 1

    return (to_rename, _resolve_chains(to_merge),
            {'collision_groups': collision_groups})


def _resolve_chains(
    pairs: List[Tuple[int, str, int, str]]
) -> List[Tuple[int, str, int, str]]:
    """Resolve duplicate→canonical chains so each merge target is stable.

    If A→B and B→C are both candidate merges, deleting B before merging A
    would create a dangling reference; walk the chain so A merges directly
    into C. Since `normalize_url` is idempotent a merge target can never be
    a duplicate itself — kept as defense in depth. Drops cycles.
    """
    direct = {d: (c, cu) for d, _du, c, cu in pairs}
    dup_url = {d: du for d, du, _c, _cu in pairs}
    dup_ids = set(direct.keys())

    resolved = []
    for dup_id in direct:
        seen = {dup_id}
        canon_id, canon_url = direct[dup_id]
        while canon_id in dup_ids:
            if canon_id in seen:  # cycle
                canon_id = None
                break
            seen.add(canon_id)
            canon_id, canon_url = direct[canon_id]
        if canon_id is None or canon_id == dup_id:
            continue
        resolved.append((dup_id, dup_url[dup_id], canon_id, canon_url))
    return resolved


def _rename_one(expr: model.Expression, new_url: str, reset_status: bool) -> None:
    """Rename an Expression in place to its canonical URL.

    Updates `domain` because the new URL may have a different host, and
    fills `original_url` for provenance.
    """
    raw_url = expr.url
    domain_name = core.get_domain_name(new_url)
    domain, _ = model.Domain.get_or_create(name=domain_name)
    expr.url = new_url
    expr.domain = domain
    if expr.original_url is None:
        expr.original_url = raw_url
    if reset_status:
        expr.http_status = None
        expr.fetched_at = None
    expr.save()


# Content/metadata fields copied to the canonical when its own value is
# empty and the duplicate's is not. relevance is deliberately excluded:
# `land consolidate` recomputes it deterministically from the merged content.
_BACKFILL_FIELDS = ('html', 'readable', 'title', 'description', 'keywords',
                    'lang', 'published_at', 'fetched_at', 'http_status',
                    'fetch_method', 'readable_at', 'approved_at', 'seorank')


def _is_empty(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _backfill_if_empty(canonical: model.Expression,
                       duplicate: model.Expression,
                       reset_status: bool = False) -> int:
    """Fill the canonical's empty fields from the duplicate before deletion.

    Never overwrites a non-empty canonical field. With reset_status, the
    fetch-state fields stay untouched (a merge must not undo the reset
    applied to renamed expressions). (validllm, validmodel) move as a pair
    so the verdict stays traceable to the model that produced it.
    depth takes the minimum (the page is reachable at the shallowest depth
    observed). Returns the number of fields filled.
    """
    filled = 0
    skip = {'fetched_at', 'http_status'} if reset_status else set()
    for field in _BACKFILL_FIELDS:
        if field in skip:
            continue
        if _is_empty(getattr(canonical, field)) and \
                not _is_empty(getattr(duplicate, field)):
            setattr(canonical, field, getattr(duplicate, field))
            filled += 1
    if _is_empty(canonical.validllm) and not _is_empty(duplicate.validllm):
        canonical.validllm = duplicate.validllm
        canonical.validmodel = duplicate.validmodel
        filled += 1
    if duplicate.depth is not None and (canonical.depth is None
                                        or duplicate.depth < canonical.depth):
        canonical.depth = duplicate.depth
        filled += 1
    if filled:
        canonical.save()
    return filled


def _merge_one(duplicate: model.Expression,
               canonical: model.Expression,
               reset_status: bool = False) -> Dict[str, int]:
    """Remap links from duplicate → canonical, backfill the canonical's
    empty fields from the duplicate, then delete the duplicate."""
    Link = model.ExpressionLink
    remapped_in = dropped_in = remapped_out = dropped_out = 0

    # Incoming: target=duplicate → target=canonical
    incoming = list(Link.select().where(Link.target == duplicate))
    for link in incoming:
        src_id = link.source_id
        if src_id == canonical.id:
            Link.delete().where((Link.source == src_id)
                                & (Link.target == duplicate)).execute()
            dropped_in += 1
            continue
        if Link.select().where((Link.source == src_id)
                               & (Link.target == canonical)).exists():
            Link.delete().where((Link.source == src_id)
                                & (Link.target == duplicate)).execute()
            dropped_in += 1
        else:
            Link.update(target=canonical).where(
                (Link.source == src_id)
                & (Link.target == duplicate)).execute()
            remapped_in += 1

    # Outgoing: source=duplicate → source=canonical
    outgoing = list(Link.select().where(Link.source == duplicate))
    for link in outgoing:
        tgt_id = link.target_id
        if tgt_id == canonical.id:
            Link.delete().where((Link.source == duplicate)
                                & (Link.target == tgt_id)).execute()
            dropped_out += 1
            continue
        if Link.select().where((Link.source == canonical)
                               & (Link.target == tgt_id)).exists():
            Link.delete().where((Link.source == duplicate)
                                & (Link.target == tgt_id)).execute()
            dropped_out += 1
        else:
            Link.update(source=canonical).where(
                (Link.source == duplicate)
                & (Link.target == tgt_id)).execute()
            remapped_out += 1

    media_count = model.Media.select().where(
        model.Media.expression == duplicate).count()
    paragraph_count = model.Paragraph.select().where(
        model.Paragraph.expression == duplicate).count()
    tagged_count = model.TaggedContent.select().where(
        model.TaggedContent.expression == duplicate).count()

    backfilled = _backfill_if_empty(canonical, duplicate, reset_status)

    duplicate.delete_instance()  # CASCADE on Media/Paragraph/TaggedContent

    return {
        'remapped_in': remapped_in,
        'dropped_in': dropped_in,
        'remapped_out': remapped_out,
        'dropped_out': dropped_out,
        'media_lost': media_count,
        'paragraphs_lost': paragraph_count,
        'tagged_lost': tagged_count,
        'backfilled': backfilled,
    }


def normalize_land(land: model.Land,
                   dry_run: bool = False,
                   limit: int = 0,
                   reset_status: bool = False,
                   verbose: bool = False) -> Dict[str, int]:
    """Apply the URL normalization pipeline retroactively to a Land.

    Returns a dict with operation counts. When `dry_run=True`, no DB write
    occurs but the same counts are computed.
    """
    print(f'Scanning land "{land.name}" for URL normalization...', flush=True)
    to_rename, to_merge, plan_stats = _collect_pairs(land)
    print(f'  {len(to_rename)} URLs to rename, {len(to_merge)} duplicates to merge'
          f' ({plan_stats["collision_groups"]} collision groups).',
          flush=True)

    if limit:
        to_rename = to_rename[:limit]
        to_merge = to_merge[:max(0, limit - len(to_rename))]

    totals = {
        'renamed': 0,
        'promoted': 0,
        'merged': 0,
        'collision_groups': plan_stats['collision_groups'],
        'remapped_in': 0,
        'dropped_in': 0,
        'remapped_out': 0,
        'dropped_out': 0,
        'media_lost': 0,
        'paragraphs_lost': 0,
        'tagged_lost': 0,
        'backfilled': 0,
        'skipped': 0,
    }

    for expr_id, old_url, new_url, promoted in to_rename:
        label = 'PROMOTE' if promoted else 'RENAME'
        if dry_run:
            totals['renamed'] += 1
            totals['promoted'] += 1 if promoted else 0
            if verbose:
                print(f'  {label} {old_url}\n      -> {new_url}', flush=True)
            continue
        expr = model.Expression.get_or_none(model.Expression.id == expr_id)
        if expr is None:
            totals['skipped'] += 1
            continue
        try:
            with model.DB.atomic():
                _rename_one(expr, new_url, reset_status)
            totals['renamed'] += 1
            totals['promoted'] += 1 if promoted else 0
            if verbose:
                print(f'  {label} {old_url} -> {new_url}', flush=True)
        except Exception as exc:
            print(f'  ! rename failed for id={expr_id}: {exc}', flush=True)
            totals['skipped'] += 1

    for dup_id, dup_url, canon_id, canon_url in to_merge:
        if dry_run:
            totals['merged'] += 1
            if verbose:
                print(f'  MERGE {dup_url}\n      -> {canon_url}', flush=True)
            continue
        # Re-fetch by id: sees the state accumulated by previous merges
        # into the same canonical, and skips rows that disappeared.
        duplicate = model.Expression.get_or_none(model.Expression.id == dup_id)
        canonical = model.Expression.get_or_none(model.Expression.id == canon_id)
        if duplicate is None or canonical is None:
            totals['skipped'] += 1
            continue
        try:
            with model.DB.atomic():
                stats = _merge_one(duplicate, canonical, reset_status)
            totals['merged'] += 1
            for k in ('remapped_in', 'dropped_in', 'remapped_out',
                      'dropped_out', 'media_lost', 'paragraphs_lost',
                      'tagged_lost', 'backfilled'):
                totals[k] += stats[k]
            if verbose:
                print(f'  MERGE {dup_url} -> {canon_url} '
                      f'(in {stats["remapped_in"]}+{stats["dropped_in"]} / '
                      f'out {stats["remapped_out"]}+{stats["dropped_out"]})',
                      flush=True)
        except Exception as exc:
            print(f'  ! merge failed for id={dup_id}: {exc}', flush=True)
            totals['skipped'] += 1

    return totals
