"""Tests for the raw-HTML link network export (sprint fullhtml-linknetwork).

Covers:
- mwi.link_context.extract_all_links (append-only, duplicates, filtering).
- The closed-network page/domain link CSVs added to nodelinkcsv under
  --fullhtml=TRUE: the union of the editorial (ExpressionLink) graph and the
  raw-only HTML edges — weightbody/weighthtml, Gephi columns, minrel scoping.
- Same-page '#' anchor filtering (sprint R2.B): absolute URL + fragment to the
  source page must not create self-loops.
- Backward compatibility: nodelinkcsv without --fullhtml emits exactly 4 files.

The two graphs are DIFFERENT extractions (raw <a href> vs Trafilatura markdown
readable stored in ExpressionLink), so neither is a superset of the other; the
fixture exercises both raw\\mywi (footer link MyWI dropped) and mywi\\raw (an
ExpressionLink edge absent from the raw HTML).
"""
import csv
import glob
import os

import pytest

from mwi import link_context
from mwi.export import Export


# --------------------------------------------------------------------------- #
# Unit tests: extract_all_links                                               #
# --------------------------------------------------------------------------- #

class TestExtractAllLinks:
    """Append-only raw <a href> extractor (duplicates preserved)."""

    def test_duplicates_preserved(self):
        html = ('<html><body>'
                '<a href="https://x.test/a">1</a>'
                '<a href="https://x.test/a">2</a>'
                '<a href="https://x.test/b">3</a>'
                '</body></html>')
        links = link_context.extract_all_links(html, 'https://x.test/')
        assert links.count('https://x.test/a') == 2
        assert links.count('https://x.test/b') == 1
        assert len(links) == 3

    def test_relative_resolved_against_base(self):
        html = '<html><body><a href="/sub/page">x</a></body></html>'
        links = link_context.extract_all_links(html, 'https://x.test/dir/')
        assert links == ['https://x.test/sub/page']

    def test_non_http_and_skip_prefixes_filtered(self):
        html = ('<html><body>'
                '<a href="mailto:a@b.test">m</a>'
                '<a href="#frag">f</a>'
                '<a href="javascript:void(0)">j</a>'
                '<a href="tel:+33">t</a>'
                '<a href="ftp://x.test/file">ftp</a>'
                '<a href="https://ok.test/p">ok</a>'
                '</body></html>')
        links = link_context.extract_all_links(html, 'https://x.test/')
        assert links == ['https://ok.test/p']

    def test_path_case_preserved(self):
        html = '<html><body><a href="https://x.test/CamelCase/Path">x</a></body></html>'
        links = link_context.extract_all_links(html, 'https://x.test/')
        assert links == ['https://x.test/CamelCase/Path']

    def test_broken_or_empty_html_never_raises(self):
        assert link_context.extract_all_links(None, 'https://x.test/') == []
        assert link_context.extract_all_links('', 'https://x.test/') == []
        # Malformed markup must not raise — returns whatever it could parse.
        out = link_context.extract_all_links('<a href=https://x.test/p>', 'https://x.test/')
        assert isinstance(out, list)

    def test_xml_document_does_not_emit_xml_warning(self):
        # Stored HTML is sometimes actually XML (sitemap/feed); parsing it
        # must not spew XMLParsedAsHTMLWarning (benign, but alarming on stdout).
        import warnings
        xml = ('<?xml version="1.0" encoding="UTF-8"?>'
               '<feed><entry><link href="https://x.test/a"/></entry></feed>')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            link_context.extract_all_links(xml, 'https://x.test/')
        assert 'XMLParsedAsHTMLWarning' not in [w.category.__name__ for w in caught]

    def test_same_page_absolute_fragment_filtered(self):
        """Absolute URL + fragment to the source page is in-page nav -> dropped."""
        html = ('<html><body>'
                '<a href="https://x.test/p#S1">s1</a>'
                '<a href="https://x.test/p#S2">s2</a>'
                '<a href="https://x.test/other#S1">other</a>'
                '</body></html>')
        links = link_context.extract_all_links(html, 'https://x.test/p')
        assert links == ['https://x.test/other#S1']

    def test_bare_self_link_filtered(self):
        """A self-link without fragment is also in-page navigation -> dropped."""
        html = '<html><body><a href="https://x.test/p">self</a></body></html>'
        assert link_context.extract_all_links(html, 'https://x.test/p') == []


# --------------------------------------------------------------------------- #
# Fixture: a small land with stored HTML and a known MyWI graph                #
# --------------------------------------------------------------------------- #

@pytest.fixture()
def fullhtml_land(fresh_db):
    """Build a closed-network test land (sprint R2: union + '#' filtering).

    Raw html outbound <a href> per expression. E1 footer links to E3 twice
    (UPPER host + trailing slash -> raw-only edge weighthtml=2), to E5 once
    (raw-only edge weighthtml=1) and carries a same-page #sec1 anchor that
    must be filtered (no E1->E1 self-loop).
    ExpressionLink (body): E1->E2, E2->E1, E3->E1, E3->E2 -> weightbody=1.

    Expressions (relevance, html outbound <a href>):
      E1 site-a/home   rel5  -> E2 x2 (content), E3 + E5 (footer), E4 (rel0), external
      E2 site-a/article rel5 -> E1
      E3 site-b/page    rel5 -> E1
      E4 site-b/lowrel  rel0 -> E1            (excluded: relevance < minrel)
      E5 site-a/nohtml  rel5  html=None        (no raw contribution)

    MyWI ExpressionLink graph (the markdown-readable extraction):
      E1->E2, E2->E1, E3->E1  (also present in raw HTML -> in_mywi=1)
      E3->E2                  (NOT in raw HTML -> contributes to mywi\\raw)
    Note E1->E3 / E1->E5 (footer) are NOT in ExpressionLink -> in_mywi=0.

    Readable markdown (citation column ground truth):
      E1 cites E2 (exact URL) and E3 (UPPER host + trailing slash variant)
      E2 readable=None (never extracted)
      E3 cites E2 only (E3->E1 is in ExpressionLink but NOT in the readable,
         simulating the crawl BS4 fallback that fills ExpressionLink from
         raw HTML anchors)
      E5 cites E2 (readable-only link: must NOT create any edge)
    -> citation=1: E1->E2, E1->E3, E3->E2
    -> citation=0: E2->E1 (readable NULL), E3->E1, E1->E5
    """
    model = fresh_db["model"]
    controller = fresh_db["controller"]
    core = fresh_db["core"]

    name = "fullhtml_net"
    controller.LandController.create(
        core.Namespace(name=name, desc="raw link net", lang=["fr"])
    )
    land = model.Land.get(model.Land.name == name)

    d_a = model.Domain.create(name="site-a.test")
    d_b = model.Domain.create(name="site-b.test")

    def mk(url, domain, rel, html, readable=None):
        return model.Expression.create(land=land, domain=domain, url=url,
                                       relevance=rel, depth=0,
                                       http_status="200", html=html,
                                       readable=readable)

    e1_html = (
        '<html><body><article>'
        '<p>Intro <a href="https://site-a.test/article">art</a> and again '
        '<a href="https://site-a.test/article">art bis</a>.</p>'
        # absolute URL + fragment to E1 itself -> in-page nav, must be filtered
        '<p>TOC <a href="https://site-a.test/home#sec1">jump</a> and '
        'low <a href="https://site-b.test/lowrel">low</a> and '
        '<a href="https://external.test/page">ext</a>.</p>'
        '</article>'
        # two footer anchors to E3 via UPPER host + trailing slash -> raw-only
        # edge (absent from ExpressionLink), weighthtml=2, robust matching.
        # One footer anchor to E5 -> raw-only edge weighthtml=1, absent from
        # E1's readable -> citation=0.
        '<footer>'
        '<a href="https://SITE-B.test/page/">site b</a>'
        '<a href="https://SITE-B.test/page/">site b again</a>'
        '<a href="https://site-a.test/nohtml">n5</a>'
        '</footer>'
        '</body></html>'
    )
    e2_html = '<html><body><p><a href="https://site-a.test/home">home</a></p></body></html>'
    e3_html = '<html><body><p><a href="https://site-a.test/home">home</a></p></body></html>'
    e4_html = '<html><body><p><a href="https://site-a.test/home">home</a></p></body></html>'

    e1 = mk("https://site-a.test/home", d_a, 5, e1_html,
            readable="Voir [art](https://site-a.test/article) et "
                     "[site b](https://SITE-B.test/page/).")
    e2 = mk("https://site-a.test/article", d_a, 5, e2_html)
    e3 = mk("https://site-b.test/page", d_b, 5, e3_html,
            readable="Lire [article](https://site-a.test/article).")
    e4 = mk("https://site-b.test/lowrel", d_b, 0, e4_html)
    e5 = mk("https://site-a.test/nohtml", d_a, 5, None,
            readable="[art](https://site-a.test/article)")

    for s, t in [(e1, e2), (e2, e1), (e3, e1), (e3, e2)]:
        model.ExpressionLink.create(source=s, target=t)

    return {
        "land": land, "name": name, "model": model,
        "controller": controller, "core": core,
        "data_dir": str(fresh_db["data_dir"]),
        "e": {"e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5},
        "d": {"d_a": d_a.id, "d_b": d_b.id},
    }


def _edge_map(csv_path):
    """Read a *_pageslinksfullhtml.csv into {(Source, Target): row}."""
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {(int(r["Source"]), int(r["Target"])): r for r in rows}


# --------------------------------------------------------------------------- #
# Closed-network semantics (direct writer calls)                              #
# --------------------------------------------------------------------------- #

class TestClosedNetworkPageLinks:

    def _write(self, fullhtml_land, tmp_path, minrel=1):
        exp = Export('nodelinkcsv', fullhtml_land["land"], minrel, fullhtml=True)
        out = str(tmp_path / "pl.csv")
        exp._write_pageslinksfullhtml(out)
        return exp, _edge_map(out)

    def test_external_target_excluded(self, fullhtml_land, tmp_path):
        """A raw href to a domain outside the corpus never appears (closed)."""
        _, edges = self._write(fullhtml_land, tmp_path)
        urls = {r["target_url"] for r in edges.values()}
        assert not any("external.test" in u for u in urls)

    def test_content_edge_is_body(self, fullhtml_land, tmp_path):
        """An edge present in ExpressionLink -> weightbody=1, weighthtml=0."""
        e = fullhtml_land["e"]
        _, edges = self._write(fullhtml_land, tmp_path)
        for s, t in ((e["e1"], e["e2"]), (e["e2"], e["e1"])):
            row = edges[(s.id, t.id)]
            assert row["weightbody"] == "1"
            assert row["weighthtml"] == "0"

    def test_footer_edge_is_rawonly(self, fullhtml_land, tmp_path):
        """E1->E3 (footer) is in the raw HTML only -> weightbody=0, weighthtml>0."""
        e = fullhtml_land["e"]
        _, edges = self._write(fullhtml_land, tmp_path)
        row = edges[(e["e1"].id, e["e3"].id)]
        assert row["weightbody"] == "0"
        assert row["weighthtml"] == "2"

    def test_weighthtml_counts_rawonly_multiplicity(self, fullhtml_land, tmp_path):
        """Two footer <a> from E1 to E3 -> a single row with weighthtml=2."""
        e = fullhtml_land["e"]
        _, edges = self._write(fullhtml_land, tmp_path)
        assert edges[(e["e1"].id, e["e3"].id)]["weighthtml"] == "2"

    def test_same_page_anchor_no_self_loop(self, fullhtml_land, tmp_path):
        """An absolute URL + fragment to the page itself is not a link.

        E1 carries <a href="https://site-a.test/home#sec1"> (its own URL); it
        must not create an E1->E1 self-loop (sprint R2.B), and no edge at all
        is a self-loop.
        """
        e = fullhtml_land["e"]
        _, edges = self._write(fullhtml_land, tmp_path)
        assert (e["e1"].id, e["e1"].id) not in edges
        assert all(s != t for s, t in edges)

    def test_gephi_columns_with_empty_weight(self, fullhtml_land, tmp_path):
        """Edge file uses Gephi names Source/Target/Weight; Weight left empty."""
        exp = Export('nodelinkcsv', fullhtml_land["land"], 1, fullhtml=True)
        out = str(tmp_path / "pl.csv")
        exp._write_pageslinksfullhtml(out)
        with open(out, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            rows = list(reader)
        assert header[:5] == ['Source', 'Target', 'Weight',
                              'weightbody', 'weighthtml']
        assert rows and all(r["Weight"] == "" for r in rows)

    def test_minrel_excludes_low_relevance_endpoints(self, fullhtml_land, tmp_path):
        """E4 (relevance 0) is neither a source nor a target with minrel=1."""
        e = fullhtml_land["e"]
        _, edges = self._write(fullhtml_land, tmp_path, minrel=1)
        ids_seen = {s for s, _ in edges} | {t for _, t in edges}
        assert e["e4"].id not in ids_seen

    def test_robust_matching_trailing_slash_and_case(self, fullhtml_land, tmp_path):
        """A raw href differing by trailing slash / host case still maps.

        E1's footer links to E3 via 'https://SITE-B.test/page/' (upper host +
        trailing slash); the relaxed / host+path key must resolve it to the
        in-corpus E3, producing the raw-only edge E1->E3.
        """
        e = fullhtml_land["e"]
        _, edges = self._write(fullhtml_land, tmp_path)
        assert (e["e1"].id, e["e3"].id) in edges

    def test_three_way_diff_counters(self, fullhtml_land, tmp_path):
        exp, edges = self._write(fullhtml_land, tmp_path)
        stats = exp._fullhtml_stats
        # 4 qualifying sources (E1,E2,E3,E5); E5 has no html.
        assert stats["pages_total"] == 4
        assert stats["pages_with_html"] == 3
        # body edges: E1->E2, E2->E1, E3->E1, E3->E2 (ExpressionLink, non-self)
        assert stats["body_edges"] == 4
        # raw-only edges: footer E1->E3 and E1->E5 (absent from ExpressionLink)
        assert stats["rawonly_edges"] == 2
        assert stats["total_edges"] == 6
        assert len(edges) == 6


class TestCitationColumn:
    """citation = 1 iff the link appears in the source's readable markdown.

    Independent from weightbody: an ExpressionLink row can come from the
    crawl BS4 fallback (raw HTML anchors) while the readable lacks the link.
    """

    def _write(self, fullhtml_land, tmp_path, minrel=1):
        exp = Export('nodelinkcsv', fullhtml_land["land"], minrel, fullhtml=True)
        out = str(tmp_path / "pl.csv")
        exp._write_pageslinksfullhtml(out)
        return exp, _edge_map(out)

    def test_header_citation_position(self, fullhtml_land, tmp_path):
        """citation sits right after weighthtml; header[:5] is unchanged."""
        exp = Export('nodelinkcsv', fullhtml_land["land"], 1, fullhtml=True)
        out = str(tmp_path / "pl.csv")
        exp._write_pageslinksfullhtml(out)
        with open(out, encoding="utf-8") as f:
            header = csv.DictReader(f).fieldnames
        assert header[:5] == ['Source', 'Target', 'Weight',
                              'weightbody', 'weighthtml']
        assert header[5] == 'citation'

    def test_body_edge_cited_in_readable(self, fullhtml_land, tmp_path):
        """E1->E2: ExpressionLink edge whose URL is in E1's readable -> 1."""
        e = fullhtml_land["e"]
        _, edges = self._write(fullhtml_land, tmp_path)
        row = edges[(e["e1"].id, e["e2"].id)]
        assert row["weightbody"] == "1"
        assert row["citation"] == "1"

    def test_body_edge_absent_from_readable_is_zero(self, fullhtml_land, tmp_path):
        """E3->E1 is in ExpressionLink but not in E3's readable (BS4
        fallback simulation) -> citation=0 despite weightbody=1."""
        e = fullhtml_land["e"]
        _, edges = self._write(fullhtml_land, tmp_path)
        row = edges[(e["e3"].id, e["e1"].id)]
        assert row["weightbody"] == "1"
        assert row["citation"] == "0"

    def test_rawonly_edge_cited_via_url_variant(self, fullhtml_land, tmp_path):
        """E1->E3 is raw-only, but E1's readable cites E3 via an UPPER host +
        trailing slash variant -> the 3-key resolver still matches -> 1."""
        e = fullhtml_land["e"]
        _, edges = self._write(fullhtml_land, tmp_path)
        row = edges[(e["e1"].id, e["e3"].id)]
        assert row["weightbody"] == "0"
        assert row["citation"] == "1"

    def test_rawonly_edge_not_cited_is_zero(self, fullhtml_land, tmp_path):
        """E1->E5 (footer anchor) is absent from E1's readable -> 0."""
        e = fullhtml_land["e"]
        _, edges = self._write(fullhtml_land, tmp_path)
        row = edges[(e["e1"].id, e["e5"].id)]
        assert row["weightbody"] == "0"
        assert row["citation"] == "0"

    def test_null_readable_gives_zero(self, fullhtml_land, tmp_path):
        """E2 has readable=None -> every edge sourced at E2 has citation=0."""
        e = fullhtml_land["e"]
        _, edges = self._write(fullhtml_land, tmp_path)
        assert edges[(e["e2"].id, e["e1"].id)]["citation"] == "0"

    def test_readable_only_link_creates_no_edge(self, fullhtml_land, tmp_path):
        """E5 cites E2 in its readable but E5->E2 is in neither ExpressionLink
        nor the raw HTML -> no new edge is created by the citation pass."""
        e = fullhtml_land["e"]
        _, edges = self._write(fullhtml_land, tmp_path)
        assert (e["e5"].id, e["e2"].id) not in edges

    def test_stats_citation_edges(self, fullhtml_land, tmp_path):
        """citation=1: E1->E2, E1->E3, E3->E2."""
        exp, _ = self._write(fullhtml_land, tmp_path)
        assert exp._fullhtml_stats["citation_edges"] == 3


class TestClosedNetworkDomainLinks:

    def test_domain_links_in_mwi_out_mwi(self, fullhtml_land, tmp_path):
        d = fullhtml_land["d"]
        exp = Export('nodelinkcsv', fullhtml_land["land"], 1, fullhtml=True)
        exp._write_pageslinksfullhtml(str(tmp_path / "pl.csv"))
        out = str(tmp_path / "dl.csv")
        exp._write_domainlinksfullhtml(out)
        with open(out, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            rows = {(int(r["Source"]), int(r["Target"])): r for r in reader}
        assert header[:5] == ['Source', 'Target', 'Weight', 'in_mwi', 'out_mwi']
        # Inter-domain only: (b->a) from body E3->E1 & E3->E2; (a->b) from the
        # raw-only footer E1->E3 (weighthtml=2).
        assert set(rows.keys()) == {(d["d_a"], d["d_b"]), (d["d_b"], d["d_a"])}
        # site-b -> site-a carries the two editorial edges -> in_mwi=2, out_mwi=0
        assert rows[(d["d_b"], d["d_a"])]["in_mwi"] == "2"
        assert rows[(d["d_b"], d["d_a"])]["out_mwi"] == "0"
        # site-a -> site-b is the raw-only footer -> in_mwi=0, out_mwi=2
        assert rows[(d["d_a"], d["d_b"])]["in_mwi"] == "0"
        assert rows[(d["d_a"], d["d_b"])]["out_mwi"] == "2"
        # Weight column left empty (Gephi default); no intra-domain self pair.
        assert rows[(d["d_a"], d["d_b"])]["Weight"] == ""
        assert (d["d_a"], d["d_a"]) not in rows


# --------------------------------------------------------------------------- #
# Integration via the CLI controller                                          #
# --------------------------------------------------------------------------- #

class TestNodelinkcsvIntegration:

    def _glob(self, data_dir, suffix):
        return glob.glob(os.path.join(data_dir, f"export_land_*nodelinkcsv*{suffix}"))

    def test_without_flag_emits_only_four_files(self, fullhtml_land):
        ctrl, core = fullhtml_land["controller"], fullhtml_land["core"]
        data_dir = fullhtml_land["data_dir"]
        ret = ctrl.LandController.export(
            core.Namespace(name=fullhtml_land["name"], type="nodelinkcsv", minrel=1)
        )
        assert ret == 1
        assert self._glob(data_dir, "_pageslinks.csv")
        assert not self._glob(data_dir, "fullhtml.csv")

    def test_with_flag_emits_only_fullhtml_files(self, fullhtml_land):
        """--fullhtml switches networks: emit ONLY the 4 fullhtml files."""
        ctrl, core = fullhtml_land["controller"], fullhtml_land["core"]
        data_dir = fullhtml_land["data_dir"]
        ret = ctrl.LandController.export(
            core.Namespace(name=fullhtml_land["name"], type="nodelinkcsv",
                           minrel=1, fullhtml="TRUE")
        )
        assert ret == 1
        for suffix in ("_pagesnodesfullhtml.csv", "_pageslinksfullhtml.csv",
                       "_domainnodesfullhtml.csv", "_domainlinksfullhtml.csv"):
            assert self._glob(data_dir, suffix), f"missing {suffix}"
        # The base MyWI files must NOT be emitted under the flag.
        assert not self._glob(data_dir, "_pageslinks.csv")
        assert not self._glob(data_dir, "_domainlinks.csv")

    def test_pagesnodesfullhtml_matches_base_pagesnodes(self, fullhtml_land):
        """Closed network -> node set identical to the base pagesnodes file.

        The flag now switches networks, so the base file comes from a
        separate export without --fullhtml.
        """
        ctrl, core = fullhtml_land["controller"], fullhtml_land["core"]
        data_dir = fullhtml_land["data_dir"]
        ctrl.LandController.export(
            core.Namespace(name=fullhtml_land["name"], type="nodelinkcsv", minrel=1)
        )
        ctrl.LandController.export(
            core.Namespace(name=fullhtml_land["name"], type="nodelinkcsv",
                           minrel=1, fullhtml="TRUE")
        )

        def ids(path):
            with open(path, encoding="utf-8") as f:
                return sorted(int(r["id"]) for r in csv.DictReader(f))

        base = self._glob(data_dir, "_pagesnodes.csv")[-1]
        full = self._glob(data_dir, "_pagesnodesfullhtml.csv")[-1]
        assert ids(base) == ids(full)


class TestLandWithoutStoredHtml:

    def test_empty_raw_network_header_only_no_crash(self, fresh_db, tmp_path):
        """All qualifying pages have html=None -> header-only file, no crash."""
        model = fresh_db["model"]
        controller = fresh_db["controller"]
        core = fresh_db["core"]
        controller.LandController.create(
            core.Namespace(name="no_html_land", desc="x", lang=["fr"])
        )
        land = model.Land.get(model.Land.name == "no_html_land")
        dom = model.Domain.create(name="z.test")
        model.Expression.create(land=land, domain=dom, url="https://z.test/a",
                                relevance=5, depth=0, http_status="200", html=None)

        exp = Export('nodelinkcsv', land, 1, fullhtml=True)
        out = str(tmp_path / "pl.csv")
        n = exp._write_pageslinksfullhtml(out)
        assert n == 0
        assert exp._fullhtml_stats["pages_with_html"] == 0
        with open(out, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows == [['Source', 'Target', 'Weight', 'weightbody',
                         'weighthtml', 'citation',
                         'source_url', 'source_domain_id',
                         'target_url', 'target_domain_id']]
