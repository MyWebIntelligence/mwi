"""
Semantic similarity pipeline (ANN + Cross-Encoder NLI) to classify relations
between paragraphs: entailment (1), neutral (0), contradiction (-1).

Dependencies on FAISS, sentence-transformers, transformers are optional.
Falls back to brute-force recall and cosine-based heuristic classification.
"""
from __future__ import annotations

import os
import json
import math
import time
from typing import List, Tuple, Dict, Optional, Callable, Set

import settings
from . import model


def _load_vectors_for_land(land: model.Land, minrel: Optional[int] = None) -> List[Tuple[int, int, List[float], str]]:
    """Load (paragraph_id, expression_id, vector, text) for a land."""
    rows = (model.Paragraph
            .select(
                model.Paragraph.id,
                model.Paragraph.expression,
                model.Paragraph.text,
                model.ParagraphEmbedding.embedding,
            )
            .join(model.Expression)
            .switch(model.Paragraph)
            .join(model.ParagraphEmbedding, on=(model.ParagraphEmbedding.paragraph == model.Paragraph.id))
            .where(model.Expression.land == land)
            )
    if isinstance(minrel, int) and minrel > 0:
        rows = rows.where(model.Expression.relevance >= minrel)

    data: List[Tuple[int, int, List[float], str]] = []
    for r in rows.iterator():
        try:
            vec = json.loads(r.paragraphembedding.embedding)  # type: ignore[attr-defined]
            # normalize to unit length for cosine/IP indexing
            n = math.sqrt(sum(x * x for x in vec)) or 1.0
            vec = [x / n for x in vec]
        except Exception:
            continue
        data.append((r.id, r.expression.id, vec, r.text))  # type: ignore[attr-defined]
    return data


class SimilarityIndex:
    def __init__(self, dim: int):
        self.dim = dim

    def add_items(self, vectors: List[List[float]]):
        raise NotImplementedError

    def query(self, vector: List[float], top_k: int) -> Tuple[List[int], List[float]]:
        raise NotImplementedError


class BruteForceIndex(SimilarityIndex):
    def __init__(self, dim: int, vectors: List[List[float]]):
        super().__init__(dim)
        self.vectors = vectors

    def add_items(self, vectors: List[List[float]]):
        self.vectors.extend(vectors)

    def query(self, vector: List[float], top_k: int) -> Tuple[List[int], List[float]]:
        sims = []
        for i, v in enumerate(self.vectors):
            # cosine since vectors normalized
            sims.append((i, sum(a * b for a, b in zip(vector, v))))
        sims.sort(key=lambda x: x[1], reverse=True)
        ids = [i for i, _ in sims[:top_k]]
        scores = [s for _, s in sims[:top_k]]
        return ids, scores


def _try_faiss(dim: int, vectors: List[List[float]]) -> Optional[SimilarityIndex]:
    try:
        import faiss  # type: ignore
    except Exception:
        return None
    # Use IP on normalized vectors = cosine
    index = faiss.IndexFlatIP(dim)
    import numpy as np
    arr = np.array(vectors, dtype='float32')
    index.add(arr)

    class _FaissIndex(SimilarityIndex):
        def __init__(self):
            super().__init__(dim)
            self.index = index

        def add_items(self, vectors: List[List[float]]):
            self.index.add(np.array(vectors, dtype='float32'))

        def query(self, vector: List[float], top_k: int) -> Tuple[List[int], List[float]]:
            D, I = self.index.search(np.array([vector], dtype='float32'), top_k)
            ids = I[0].tolist()
            scores = D[0].tolist()
            return ids, scores

    return _FaissIndex()


def _get_index(backend: str, dim: int, vectors: List[List[float]]) -> SimilarityIndex:
    b = (backend or settings.similarity_backend or 'bruteforce').lower()
    if b == 'faiss':
        idx = _try_faiss(dim, vectors)
        if idx:
            return idx
    return BruteForceIndex(dim, vectors)


_nli_predictor: Optional[Callable[[List[Tuple[str, str]]], List[Tuple[int, float]]]] = None
_nli_backend: str = ""


def _get_nli_predictor() -> Callable[[List[Tuple[str, str]]], List[Tuple[int, float]]]:
    """Create and cache an NLI predictor. Loads model only once.
    Returns a callable mapping list[(premise, hypothesis)] -> list[(relation, confidence)].
    relation in {1 (entailment), 0 (neutral), -1 (contradiction)}
    """
    global _nli_predictor, _nli_backend
    if _nli_predictor is not None:
        return _nli_predictor

    name = settings.nli_model_name

    backend_pref = str(getattr(settings, 'nli_backend_preference', 'auto') or 'auto').lower()

    # Try sentence-transformers CrossEncoder first, unless preference skips it
    if backend_pref in ('auto', 'st', 'crossencoder'):
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            print(f"Loading NLI CrossEncoder model: {name} (first run may download weights)…", flush=True)
            ce = CrossEncoder(name, num_labels=3)

            def _predict_ce(pairs: List[Tuple[str, str]]) -> List[Tuple[int, float]]:
                probs = ce.predict(pairs, apply_softmax=True)  # shape [N,3]
                out: List[Tuple[int, float]] = []
                if hasattr(ce, 'model') and hasattr(ce.model, 'config') and hasattr(ce.model.config, 'id2label'):
                    id2label = ce.model.config.id2label  # type: ignore[attr-defined]
                else:
                    id2label = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}
                import numpy as np
                for p in probs:
                    idx = int(np.argmax(p))
                    label = str(id2label.get(idx, 'neutral')).lower()
                    conf = float(p[idx])
                    rel = 0
                    if 'entail' in label:
                        rel = 1
                    elif 'contradict' in label:
                        rel = -1
                    out.append((rel, conf))
                return out

            _nli_predictor = _predict_ce
            _nli_backend = 'sentence-transformers'
            print("NLI model ready via sentence-transformers.", flush=True)
            return _nli_predictor
        except Exception as e:
            print(f"CrossEncoder unavailable ({e}); falling back to transformers.", flush=True)

    # Try transformers AutoModel
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
        from transformers.utils import logging as hf_logging  # type: ignore
        import torch  # type: ignore
        # Environment safety knobs for macOS / BLAS / tokenizers
        try:
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        except Exception:
            pass

        # Pre-check sentencepiece requirement for some models (e.g., DeBERTa, XLM-R, ALBERT)
        lname = (name or "").lower()
        requires_spm = any(tok in lname for tok in ("deberta", "xlm", "albert"))
        if requires_spm:
            try:
                import sentencepiece  # type: ignore  # noqa: F401
            except Exception:
                fallback = getattr(settings, 'nli_fallback_model_name', 'typeform/distilbert-base-uncased-mnli')
                print(f"SentencePiece manquant pour le modèle '{name}'. Bascule automatique vers '{fallback}'.", flush=True)
                name = fallback

        print(f"Loading NLI transformers model: {name} (first run may download weights)…", flush=True)
        # Force slow tokenizer to avoid Rust tokenizers segfaults on some macOS setups
        tok = AutoTokenizer.from_pretrained(name, use_fast=False)
        mdl = AutoModelForSequenceClassification.from_pretrained(name)
        # Force CPU to avoid MPS/GPU quirks on macOS
        device = 'cpu'
        try:
            threads = int(getattr(settings, 'nli_torch_num_threads', 1) or 1)
            torch.set_num_threads(max(1, threads))
            if hasattr(torch, 'set_num_interop_threads'):
                torch.set_num_interop_threads(max(1, threads))
            os.environ.setdefault("OMP_NUM_THREADS", str(max(1, threads)))
            os.environ.setdefault("MKL_NUM_THREADS", str(max(1, threads)))
        except Exception:
            pass
        mdl.to(device)
        mdl.eval()
        # Reduce HF warning noise (overflow tokens warnings, etc.)
        try:
            hf_logging.set_verbosity_error()
        except Exception:
            pass

        max_len = int(getattr(settings, 'nli_max_tokens', 512) or 512)

        def _predict_hf(pairs: List[Tuple[str, str]]) -> List[Tuple[int, float]]:
            out: List[Tuple[int, float]] = []
            with torch.no_grad():
                for a, b in pairs:
                    inputs = tok(
                        a,
                        b,
                        return_tensors='pt',
                        truncation=True,
                        padding=True,
                        max_length=max_len,
                        return_overflowing_tokens=False,
                    )
                    # keep tensors on CPU
                    logits = mdl(**inputs).logits[0]
                    probs = torch.softmax(logits, dim=-1).tolist()
                    idx = int(max(range(len(probs)), key=lambda i: probs[i]))
                    label = str(mdl.config.id2label.get(idx, 'neutral')).lower()
                    conf = float(probs[idx])
                    rel = 0
                    if 'entail' in label:
                        rel = 1
                    elif 'contradict' in label:
                        rel = -1
                    out.append((rel, conf))
            return out

        _nli_predictor = _predict_hf
        _nli_backend = 'transformers'
        print("NLI model ready via transformers (CPU).", flush=True)
        return _nli_predictor
    except Exception as e:
        print(f"Transformers NLI unavailable ({e}); using neutral fallback.", flush=True)

    # Fallback heuristic: mark all as neutral
    def _predict_fallback(pairs: List[Tuple[str, str]]) -> List[Tuple[int, float]]:
        return [(0, 0.5) for _ in pairs]

    _nli_predictor = _predict_fallback
    _nli_backend = 'fallback'
    return _nli_predictor


def run_semantic_similarity(
    land: model.Land,
    backend: Optional[str] = None,
    top_k: Optional[int] = None,
    minrel: Optional[int] = None,
    max_pairs: Optional[int] = None,
) -> int:
    """Recall via ANN + NLI classification. Stores results in ParagraphSimilarity (method='nli').
    Returns number of pairs written.
    """
    data = _load_vectors_for_land(land, minrel=minrel)
    if not data:
        return 0
    pid_list = [pid for pid, _, _, _ in data]
    expr_list = [eid for _, eid, _, _ in data]
    text_list = [txt for _, _, _, txt in data]
    vec_list = [vec for _, _, vec, _ in data]
    dim = len(vec_list[0])
    top_k = int(top_k or settings.similarity_top_k)
    chosen_backend = (backend or settings.similarity_backend)
    print(
        f"\n🚀 Starting NLI semantic similarity: land={land.name}, paragraphs={len(pid_list)}, top_k={top_k}, minrel={minrel or 0}, backend={chosen_backend}",
        flush=True,
    )
    idx = _get_index(chosen_backend, dim, vec_list)
    print("🔎 ANN index ready.", flush=True)

    # Remove existing 'nli' similarities for these paragraphs to avoid duplicates
    if pid_list:
        (model.ParagraphSimilarity
         .delete()
         .where((model.ParagraphSimilarity.source_paragraph.in_(pid_list)) &
                (model.ParagraphSimilarity.method == 'nli'))
         ).execute()

    # Build candidate pairs (i<j to avoid duplicates)
    pairs_idx: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()
    for i, (pid_i, expr_i, vec_i, _) in enumerate(data):
        neighbor_ids, _ = idx.query(vec_i, top_k + 1)  # includes self at rank 0
        added = 0
        for nbr in neighbor_ids:
            if nbr == i:
                continue
            pid_j, expr_j, _, _ = data[nbr]
            if expr_i == expr_j:
                continue
            a, b = (pid_i, pid_j) if pid_i < pid_j else (pid_j, pid_i)
            if a == b:
                continue
            if (a, b) not in seen:
                pairs_idx.append((a, b))
                seen.add((a, b))
                added += 1
                if added >= top_k:
                    break
        if (i + 1) % 500 == 0:
            print(f"  • Recall progress: {i + 1}/{len(data)} paragraphs, candidate pairs={len(pairs_idx)}", flush=True)
        if max_pairs and len(pairs_idx) >= max_pairs:
            break

    print(f"📌 Candidate pairs ready: {len(pairs_idx)}", flush=True)
    if not pairs_idx:
        return 0

    # Prepare texts for NLI
    text_by_pid: Dict[int, str] = {pid: text for pid, _, _, text in data}
    batch_size = int(getattr(settings, 'nli_batch_size', 64) or 64)
    total = 0
    to_insert: List[Dict] = []
    predictor = _get_nli_predictor()
    if _nli_backend:
        print(f"🧠 NLI backend: {_nli_backend}", flush=True)
    print(f"⚖️ Scoring with NLI in batches of {batch_size}…", flush=True)
    total_pairs = len(pairs_idx)
    report_every = int(getattr(settings, 'nli_progress_every_pairs', 1000) or 1000)
    last_report = 0
    t0 = time.time()
    # Batch NLI
    for i in range(0, len(pairs_idx), batch_size):
        batch_pairs_idx = pairs_idx[i:i + batch_size]
        pairs_text = [(text_by_pid[a], text_by_pid[b]) for a, b in batch_pairs_idx]
        preds = predictor(pairs_text)
        for (a, b), (rel, conf) in zip(batch_pairs_idx, preds):
            to_insert.append({
                'source_paragraph': a,
                'target_paragraph': b,
                'score': float(rel),
                'score_raw': float(conf),
                'method': 'nli',
            })
            total += 1
        if len(to_insert) >= 5000:
            _flush_similarities(to_insert)
            to_insert = []
        processed = min(i + batch_size, total_pairs)
        if processed - last_report >= report_every:
            if getattr(settings, 'nli_show_throughput', True):
                dt = max(1e-6, time.time() - t0)
                pps = processed / dt
                remain = max(0, total_pairs - processed)
                eta = remain / pps if pps > 0 else float('inf')
                pct = 100.0 * processed / max(1, total_pairs)
                print(f"  • NLI progress: {processed}/{total_pairs} ({pct:.1f}%), {pps:.1f} pairs/s, ETA {eta:.0f}s", flush=True)
            else:
                print(f"  • NLI progress: {processed}/{total_pairs}", flush=True)
            last_report = processed
        if max_pairs and total >= max_pairs:
            break

    _flush_similarities(to_insert)
    # Final summary
    dt = time.time() - t0
    if total > 0 and dt > 0:
        print(f"✅ NLI scoring complete: {total} pairs, {dt:.1f}s, {total/dt:.1f} pairs/s", flush=True)
    return total


def _flush_similarities(rows: List[Dict]):
    if not rows:
        return
    # Backward-compat: older DBs may miss 'score_raw'. If so, drop it on insert.
    # Cache the check to avoid repeated PRAGMA calls.
    global _PS_HAS_SCORE_RAW, _WARNED_NO_SCORE_RAW
    try:
        _PS_HAS_SCORE_RAW
    except NameError:
        _PS_HAS_SCORE_RAW = None  # type: ignore
        _WARNED_NO_SCORE_RAW = False  # type: ignore

    if _PS_HAS_SCORE_RAW is None:
        try:
            cols = [row[1] for row in model.DB.execute_sql('PRAGMA table_info(paragraph_similarity)').fetchall()]
            _PS_HAS_SCORE_RAW = ('score_raw' in cols)  # type: ignore
        except Exception:
            _PS_HAS_SCORE_RAW = True  # safe default: assume present

    ins_rows = rows
    if not _PS_HAS_SCORE_RAW:
        # Drop the unsupported key
        ins_rows = [{k: v for k, v in r.items() if k != 'score_raw'} for r in rows]
        if not _WARNED_NO_SCORE_RAW:
            print("[warning] paragraph_similarity.score_raw missing; run 'python mywi.py db migrate' to add it. Inserting without confidence.", flush=True)
            _WARNED_NO_SCORE_RAW = True  # type: ignore

    with model.DB.atomic():
        try:
            model.ParagraphSimilarity.insert_many(ins_rows).execute()
        except Exception:
            for r in ins_rows:
                try:
                    model.ParagraphSimilarity.create(**r)
                except Exception:
                    pass
