"""
Embedding pipeline: paragraph splitting, embeddings generation, similarity computation.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import List, Tuple, Dict, DefaultDict

import requests

import settings
from . import model


def _clean_text(s: str) -> str:
    """Clean and normalize text by collapsing whitespace.

    Args:
        s: Input text string to clean.

    Returns:
        Normalized string with collapsed whitespace and trimmed edges.
    """
    return re.sub(r"\s+", " ", (s or "").strip())


def split_into_paragraphs(expression: model.Expression) -> List[str]:
    """Split expression.readable into normalized paragraphs.

    Args:
        expression: Expression object containing readable text content.

    Returns:
        List of normalized paragraph strings that meet length criteria.

    Rules:
        - Split on blank lines (two or more newlines)
        - Trim whitespace, collapse internal spaces
        - Filter by min/max length from settings
        - Falls back to whole text if no paragraphs meet criteria
    """
    if not expression.readable:
        return []
    text = expression.readable
    # Split on multiple newlines
    raw_parts = re.split(r"\n\s*\n+", text)

    parts: List[str] = []
    for chunk in raw_parts:
        t = _clean_text(chunk)
        if not t:
            continue
        if len(t) < settings.embed_min_paragraph_chars:
            continue
        if len(t) > settings.embed_max_paragraph_chars:
            continue
        parts.append(t)
    # Fallback: if nothing passed the filters, take the whole cleaned text (trim to max)
    if not parts:
        whole = _clean_text(text)
        if len(whole) > settings.embed_max_paragraph_chars:
            whole = whole[:settings.embed_max_paragraph_chars]
        # Force at least one paragraph if any content exists, even if below min length
        if len(whole) > 0:
            parts = [whole]
    return parts


def _fake_embed(texts: List[str]) -> List[List[float]]:
    """Generate deterministic local embeddings for testing/offline mode.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of normalized embedding vectors (64 dimensions).

    Note:
        Maps characters to hash-based buckets and normalizes to unit length.
        Deterministic for testing purposes.
    """
    dim = 64
    vecs: List[List[float]] = []
    for t in texts:
        arr = [0.0] * dim
        for ch in t.lower():
            idx = (ord(ch) * 1315423911) % dim
            arr[idx] += 1.0
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in arr)) or 1.0
        vecs.append([v / norm for v in arr])
    return vecs


def _http_embed(texts: List[str]) -> List[List[float]]:
    """Call a generic HTTP embedding API.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors from the API.

    Raises:
        RuntimeError: If embed_api_url is not configured.
        requests.HTTPError: If the API request fails.

    Note:
        Expects JSON response: {"data": [{"embedding": [...]}, ...]}.
    """
    if not settings.embed_api_url:
        raise RuntimeError("embed_api_url is not configured")
    payload = {"model": settings.embed_model_name, "input": texts}
    headers = {}
    # allow custom headers mapping from settings
    try:
        if isinstance(settings.embed_http_headers, str):
            headers = json.loads(settings.embed_http_headers)  # type: ignore
        elif isinstance(settings.embed_http_headers, dict):
            headers = dict(settings.embed_http_headers)
    except Exception:
        headers = {}
    resp = requests.post(settings.embed_api_url, json=payload, headers=headers, timeout=settings.default_timeout)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("data") or []
    return [it.get("embedding", []) for it in items]


def _openai_embed(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using OpenAI API.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors from OpenAI.

    Raises:
        requests.HTTPError: If the API request fails.
    """
    base = (settings.embed_openai_base_url or "https://api.openai.com/v1").rstrip("/")
    url = f"{base}/embeddings"
    headers = {
        "Authorization": f"Bearer {settings.embed_openai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": settings.embed_model_name, "input": texts}
    resp = requests.post(url, json=payload, headers=headers, timeout=settings.default_timeout)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    return [item.get("embedding", []) for item in data]


def _mistral_embed(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using Mistral AI API.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors from Mistral AI.

    Raises:
        requests.HTTPError: If the API request fails.
    """
    base = (settings.embed_mistral_base_url or "https://api.mistral.ai/v1").rstrip("/")
    url = f"{base}/embeddings"
    headers = {
        "Authorization": f"Bearer {settings.embed_mistral_api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": settings.embed_model_name, "input": texts}
    resp = requests.post(url, json=payload, headers=headers, timeout=settings.default_timeout)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    return [item.get("embedding", []) for item in data]


def _gemini_embed(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using Google Gemini API.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors from Gemini.

    Raises:
        requests.HTTPError: If the API request fails.

    Note:
        Uses batchEmbedContents endpoint for multiple inputs.
    """
    # Use batchEmbedContents for multiple inputs
    base = (settings.embed_gemini_base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
    model_name = settings.embed_model_name
    # If caller passes short model name, prepend 'models/'
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"
    url = f"{base}/{model_name}:batchEmbedContents?key={settings.embed_gemini_api_key}"
    reqs = []
    for t in texts:
        reqs.append({
            "model": model_name,
            "content": {"parts": [{"text": t}]}
        })
    payload = {"requests": reqs}
    resp = requests.post(url, json=payload, timeout=settings.default_timeout)
    resp.raise_for_status()
    embeddings = resp.json().get("embeddings", [])
    # each item: {"values": [..]}
    return [item.get("values", []) for item in embeddings]


def _huggingface_embed(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using HuggingFace Inference API.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors from HuggingFace.

    Raises:
        requests.HTTPError: If the API request fails.

    Note:
        Normalizes various response formats from HuggingFace API.
    """
    base = (settings.embed_hf_base_url or "https://api-inference.huggingface.co/models").rstrip("/")
    model_name = settings.embed_model_name
    url = f"{base}/{model_name}"
    headers = {
        "Authorization": f"Bearer {settings.embed_hf_api_key}",
        "Content-Type": "application/json",
    }
    payload = {"inputs": texts}
    resp = requests.post(url, json=payload, headers=headers, timeout=settings.default_timeout)
    resp.raise_for_status()
    data = resp.json()
    # HF returns list for single/ multi: we normalize to list-of-vectors
    # Possible shapes: [vec] or [[vec], [vec], ...] or nested. Flatten one level when needed.
    if isinstance(data, list) and data and isinstance(data[0], list) and isinstance(data[0][0], (int, float)):
        return data  # already list-of-vectors
    if isinstance(data, list) and data and isinstance(data[0], list) and isinstance(data[0][0], list):
        return [v for v in data]  # assume list-of-vectors
    if isinstance(data, list) and data and isinstance(data[0], (int, float)):
        return [data]
    return []


def _ollama_embed(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using Ollama local API.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors from Ollama.

    Raises:
        requests.HTTPError: If the API request fails.

    Note:
        Processes texts sequentially as Ollama API accepts one text per call.
    """
    base = (settings.embed_ollama_base_url or "http://localhost:11434").rstrip("/")
    url = f"{base}/api/embeddings"
    out: List[List[float]] = []
    for t in texts:
        payload = {"model": settings.embed_model_name, "prompt": t}
        resp = requests.post(url, json=payload, timeout=settings.default_timeout)
        resp.raise_for_status()
        vec = resp.json().get("embedding", [])
        out.append(vec)
    return out


def _embed_texts(texts: List[str]) -> List[List[float]]:
    """Route embedding requests to the configured provider.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors from the configured provider.

    Note:
        Provider is determined by settings.embed_provider.
        Supported providers: fake, http, openai, mistral, gemini,
        huggingface, ollama. Falls back to fake for unknown providers.
    """
    provider = (settings.embed_provider or 'fake').lower()
    if provider == 'fake':
        return _fake_embed(texts)
    elif provider == 'http':
        return _http_embed(texts)
    elif provider == 'openai':
        return _openai_embed(texts)
    elif provider == 'mistral':
        return _mistral_embed(texts)
    elif provider == 'gemini':
        return _gemini_embed(texts)
    elif provider == 'huggingface':
        return _huggingface_embed(texts)
    elif provider == 'ollama':
        return _ollama_embed(texts)
    else:
        # Fallback to fake for unknown provider
        return _fake_embed(texts)


def generate_embeddings_for_paragraphs(land: model.Land, limit_expressions: int | None = None) -> Tuple[int, int]:
    """Create Paragraph rows from readable content and generate embeddings.

    Args:
        land: Land object containing expressions to process.
        limit_expressions: Optional limit on number of expressions to process.

    Returns:
        Tuple of (paragraphs_created, embeddings_created) counts.

    Note:
        Creates Paragraph entries with text hash deduplication, then
        generates embeddings for paragraphs missing them in batches.
    """
    # Select expressions of the land with readable content
    expr_query = model.Expression.select().where(
        (model.Expression.land == land) & (model.Expression.readable.is_null(False))
    )
    if limit_expressions and limit_expressions > 0:
        expr_query = expr_query.limit(limit_expressions)

    paragraphs_created = 0
    embeddings_created = 0

    # Create Paragraph rows
    for expr in expr_query:
        paras = split_into_paragraphs(expr)
        pidx = 0
        for ptxt in paras:
            pidx += 1
            th = hashlib.sha256(ptxt.encode('utf-8')).hexdigest()
            # Deduplicate globally by text hash
            paragraph, created = model.Paragraph.get_or_create(
                text_hash=th,
                defaults={
                    'expression': expr,
                    'domain': expr.domain,
                    'para_index': pidx,
                    'text': ptxt,
                }
            )
            if created:
                paragraphs_created += 1

    # Generate embeddings for paragraphs missing them
    # Collect paragraphs without embedding for this land
    q = (model.Paragraph
         .select(model.Paragraph.id, model.Paragraph.text)
         .join(model.Expression)
         .where((model.Expression.land == land) &
                (~model.Paragraph.id.in_(
                    model.ParagraphEmbedding.select(model.ParagraphEmbedding.paragraph)
                )))
         )
    batch = []
    batch_ids = []
    for row in q.iterator():
        batch.append(row.text)
        batch_ids.append(row.id)
        if len(batch) >= settings.embed_batch_size:
            embeddings_created += _persist_embeddings(batch_ids, batch)
            batch, batch_ids = [], []
    if batch:
        embeddings_created += _persist_embeddings(batch_ids, batch)

    return paragraphs_created, embeddings_created


def _persist_embeddings(ids: List[int], texts: List[str]) -> int:
    """Generate and persist embeddings for paragraph batch.

    Args:
        ids: List of paragraph IDs.
        texts: List of corresponding text strings.

    Returns:
        Number of embedding records created.

    Note:
        Computes L2 norm and stores embeddings as JSON in database.
    """
    vecs = _embed_texts(texts)
    created = 0
    with model.DB.atomic():
        for pid, vec in zip(ids, vecs):
            # Compute norm if not normalized
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            payload = json.dumps(vec)
            if not model.ParagraphEmbedding.get_or_none(model.ParagraphEmbedding.paragraph == pid):
                model.ParagraphEmbedding.create(
                    paragraph=pid,
                    embedding=payload,
                    norm=norm,
                    model_name=settings.embed_model_name,
                )
                created += 1
    return created


def _cosine(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        Cosine similarity score between -1 and 1.

    Note:
        Normalizes vectors during computation if not already normalized.
    """
    # a and b expected normalized or will be normalized during compute
    dot = sum(x * y for x, y in zip(a, b))
    # in case they are not normalized
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def compute_paragraph_similarities(
    land: model.Land,
    threshold: float | None = None,
    method: str | None = None,
    top_k: int | None = None,
    lsh_bits: int | None = None,
    minrel: int | None = None,
    max_pairs: int | None = None,
) -> int:
    """Compute pairwise cosine similarities and store pairs above threshold.

    Args:
        land: Land object containing paragraphs to compare.
        threshold: Minimum similarity score to store (default from settings).
        method: Similarity computation method (default from settings).
        top_k: Optional limit on similarity pairs per source paragraph.
        lsh_bits: Number of LSH bits for cosine_lsh method.
        minrel: Minimum expression relevance filter.
        max_pairs: Maximum total similarity pairs to compute.

    Returns:
        Number of similarity pairs inserted.

    Note:
        Clears existing similarities for the land before computing.
        Supports 'cosine' and 'cosine_lsh' methods.
    """
    thr = float(threshold if threshold is not None else settings.embed_similarity_threshold)
    meth = (method or settings.embed_similarity_method or 'cosine')

    # Load paragraphs with embeddings for this land (optionally filter by min relevance)
    rows = (model.Paragraph
            .select(
                model.Paragraph.id,
                model.Paragraph.expression,
                model.ParagraphEmbedding.embedding,
            )
            .join(model.Expression)
            .switch(model.Paragraph)
            .join(model.ParagraphEmbedding, on=(model.ParagraphEmbedding.paragraph == model.Paragraph.id))
            .where(model.Expression.land == land)
            )
    if isinstance(minrel, int) and minrel > 0:
        rows = rows.where(model.Expression.relevance >= minrel)
    data: List[Tuple[int, int, List[float]]] = []
    for r in rows.iterator():
        try:
            vec = json.loads(r.paragraphembedding.embedding)  # type: ignore[attr-defined]
        except Exception:
            continue
        data.append((r.id, r.expression.id, vec))  # type: ignore[attr-defined]

    # Clear existing similarities for this land+method to avoid duplicates
    land_para_ids = [pid for pid, _, _ in data]
    if land_para_ids:
        model.ParagraphSimilarity.delete().where(
            (model.ParagraphSimilarity.source_paragraph.in_(land_para_ids)) &
            (model.ParagraphSimilarity.method == meth)
        ).execute()

    # Select compute strategy
    algo = (meth or 'cosine').lower()
    if algo == 'cosine_lsh':
        return _compute_similarities_lsh(data, thr, meth, top_k=top_k, lsh_bits=lsh_bits or 16, max_pairs=max_pairs)
    else:
        return _compute_similarities_bruteforce(data, thr, meth, top_k=top_k, max_pairs=max_pairs)


def _compute_similarities_bruteforce(
    data: List[Tuple[int, int, List[float]]], thr: float, meth: str,
    top_k: int | None, max_pairs: int | None
) -> int:
    """Compute similarities using brute-force pairwise comparison.

    Args:
        data: List of (paragraph_id, expression_id, embedding_vector) tuples.
        thr: Minimum similarity threshold.
        meth: Method name to store.
        top_k: Optional limit on pairs per source paragraph.
        max_pairs: Optional maximum total pairs limit.

    Returns:
        Number of similarity pairs created.
    """
    n = len(data)
    count = 0
    batch_inserts = []
    # Optional top-k limiter per source paragraph
    for i in range(n):
        pid_i, expr_i, vec_i = data[i]
        candidates: List[Tuple[float, int]] = []  # (score, target_pid)
        for j in range(i + 1, n):
            pid_j, expr_j, vec_j = data[j]
            if expr_i == expr_j:
                continue
            score = _cosine(vec_i, vec_j)
            if score >= thr:
                candidates.append((score, pid_j))
        if top_k and len(candidates) > top_k:
            candidates.sort(key=lambda x: x[0], reverse=True)
            candidates = candidates[:top_k]
        for score, pid_j in candidates:
            batch_inserts.append({
                'source_paragraph': pid_i,
                'target_paragraph': pid_j,
                'score': score,
                'method': meth,
            })
            count += 1
            if max_pairs and count >= max_pairs:
                _flush_similarity_inserts(batch_inserts)
                return count
        # Flush periodically to reduce memory
        if len(batch_inserts) >= 5000:
            _flush_similarity_inserts(batch_inserts)
            batch_inserts = []

    _flush_similarity_inserts(batch_inserts)
    return count


def _compute_similarities_lsh(
    data: List[Tuple[int, int, List[float]]], thr: float, meth: str,
    top_k: int | None, lsh_bits: int, max_pairs: int | None
) -> int:
    """Compute similarities using LSH (Locality-Sensitive Hashing).

    Args:
        data: List of (paragraph_id, expression_id, embedding_vector) tuples.
        thr: Minimum similarity threshold.
        meth: Method name to store.
        top_k: Optional limit on pairs per source paragraph.
        lsh_bits: Number of hash bits for bucketization.
        max_pairs: Optional maximum total pairs limit.

    Returns:
        Number of similarity pairs created.

    Note:
        Uses random hyperplanes for LSH signatures, then performs
        brute-force comparison within each bucket.
    """
    import random
    random.seed(42)
    if not data:
        return 0
    dim = len(data[0][2])
    # Generate random hyperplanes
    planes: List[List[float]] = []
    for _ in range(lsh_bits):
        # Simple Rademacher vectors {-1, +1}
        v = [1.0 if random.random() > 0.5 else -1.0 for _ in range(dim)]
        planes.append(v)

    def signature(vec: List[float]) -> int:
        sig = 0
        for b, plane in enumerate(planes):
            # dot(vec, plane) sign
            dot = 0.0
            for x, y in zip(vec, plane):
                dot += x * y
            if dot >= 0:
                sig |= (1 << b)
        return sig

    # Bucketize by signature
    from collections import defaultdict
    buckets: DefaultDict[int, List[Tuple[int, int, List[float]]]] = defaultdict(list)
    for item in data:
        buckets[signature(item[2])].append(item)

    count = 0
    batch_inserts = []
    # For each bucket, do local brute-force with optional top-k
    for sig, items in buckets.items():
        m = len(items)
        for i in range(m):
            pid_i, expr_i, vec_i = items[i]
            candidates: List[Tuple[float, int]] = []
            for j in range(i + 1, m):
                pid_j, expr_j, vec_j = items[j]
                if expr_i == expr_j:
                    continue
                score = _cosine(vec_i, vec_j)
                if score >= thr:
                    candidates.append((score, pid_j))
            if top_k and len(candidates) > top_k:
                candidates.sort(key=lambda x: x[0], reverse=True)
                candidates = candidates[:top_k]
            for score, pid_j in candidates:
                batch_inserts.append({
                    'source_paragraph': pid_i,
                    'target_paragraph': pid_j,
                    'score': score,
                    'method': meth,
                })
                count += 1
                if max_pairs and count >= max_pairs:
                    _flush_similarity_inserts(batch_inserts)
                    return count
            if len(batch_inserts) >= 5000:
                _flush_similarity_inserts(batch_inserts)
                batch_inserts = []

    _flush_similarity_inserts(batch_inserts)
    return count


def _flush_similarity_inserts(rows: List[Dict]):
    """Persist similarity pairs to database with batch insert.

    Args:
        rows: List of dictionaries containing similarity pair data.

    Note:
        Attempts batch insert, falls back to individual inserts on error.
        Silently ignores duplicate errors.
    """
    if not rows:
        return
    with model.DB.atomic():
        try:
            model.ParagraphSimilarity.insert_many(rows).execute()
        except Exception:
            # Fallback: insert one by one ignoring duplicates
            for r in rows:
                try:
                    model.ParagraphSimilarity.create(**r)
                except Exception:
                    pass
