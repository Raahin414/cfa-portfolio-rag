import json
import os
import re
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_INDEX = os.getenv("PINECONE_INDEX", "portfolio-rag-384")
DEFAULT_NAMESPACE = "recursive"
CHUNK_FILE_MAP = {
    "fixed": "fixed_chunks.json",
    "recursive": "recursive_chunks.json",
    "semantic": "semantic_chunks.json",
}


load_dotenv()

model = SentenceTransformer(MODEL_NAME)

api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("PINECONE_API_KEY is missing. Add it to your environment or .env file.")

pc = Pinecone(api_key=api_key)
index = pc.Index(DEFAULT_INDEX)

_CACHE = {}


def _tokenize(text):
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def _extract_matches(result):
    if isinstance(result, dict):
        return result.get("matches", [])
    return getattr(result, "matches", []) or []


def _get_chunk_id(match):
    if isinstance(match, dict):
        return match.get("id", "")
    return getattr(match, "id", "")


def _get_chunk_score(match):
    if isinstance(match, dict):
        return float(match.get("score", 0.0))
    return float(getattr(match, "score", 0.0))


def _load_strategy_corpus(strategy):
    if strategy in _CACHE:
        return _CACHE[strategy]

    chunk_file = CHUNK_FILE_MAP.get(strategy)
    if not chunk_file:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {sorted(CHUNK_FILE_MAP)}")

    chunk_path = Path(chunk_file)
    if not chunk_path.exists():
        _CACHE[strategy] = {
            "data": [],
            "texts": [],
            "bm25": None,
            "local_by_id": {},
            "has_local_corpus": False,
        }
        return _CACHE[strategy]

    with chunk_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [item.get("text", "") for item in data]
    tokenized_docs = [_tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized_docs)
    local_by_id = {item.get("id", ""): item for item in data if item.get("id")}

    _CACHE[strategy] = {
        "data": data,
        "texts": texts,
        "bm25": bm25,
        "local_by_id": local_by_id,
        "has_local_corpus": True,
    }
    return _CACHE[strategy]


def _get_match_metadata(match):
    if isinstance(match, dict):
        return match.get("metadata", {}) or {}
    return getattr(match, "metadata", {}) or {}


def _normalize_scores(score_dict):
    if not score_dict:
        return {}
    values = list(score_dict.values())
    min_v = min(values)
    max_v = max(values)
    if abs(max_v - min_v) < 1e-9:
        return {k: 1.0 for k in score_dict}
    return {k: (v - min_v) / (max_v - min_v) for k, v in score_dict.items()}


def _source_multiplier(metadata):
    src = str(metadata.get("source", "") or "").lower()
    title = str(metadata.get("title", "") or "").lower()
    joined = f"{src} {title}"

    # Prefer CFA/Kaplan-style book material when available.
    if any(token in joined for token in ("cfa", "kaplan", "schweser", ".pdf", "book")):
        return 1.15

    # Keep web content available, but reduce dominance in final ranking.
    if "investopedia" in joined:
        return 0.7

    return 1.0


def hybrid_search(
    query,
    top_k=5,
    strategy=DEFAULT_NAMESPACE,
    semantic_weight=0.6,
    bm25_weight=0.4,
    return_debug=False,
):
    start = time.perf_counter()

    if semantic_weight < 0 or bm25_weight < 0:
        raise ValueError("semantic_weight and bm25_weight must be non-negative")
    if semantic_weight + bm25_weight == 0:
        raise ValueError("At least one retrieval weight must be > 0")

    corpus = _load_strategy_corpus(strategy)
    data = corpus["data"]
    bm25 = corpus["bm25"]
    local_by_id = corpus["local_by_id"]
    has_local_corpus = corpus.get("has_local_corpus", False)

    query_embedding = model.encode(query).tolist()
    semantic_raw = index.query(
        vector=query_embedding,
        top_k=max(top_k * 3, 10),
        include_metadata=True,
        namespace=strategy,
    )
    semantic_matches = _extract_matches(semantic_raw)

    semantic_scores = {}
    for m in semantic_matches:
        cid = _get_chunk_id(m)
        if cid:
            semantic_scores[cid] = max(semantic_scores.get(cid, -1e9), _get_chunk_score(m))

    bm25_scores = {}
    if bm25 is not None and has_local_corpus:
        tokenized_query = _tokenize(query)
        raw_bm25_scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(raw_bm25_scores)[-max(top_k * 3, 10):][::-1]

        for idx in top_indices:
            cid = data[idx].get("id", "")
            if cid:
                bm25_scores[cid] = float(raw_bm25_scores[idx])

    semantic_norm = _normalize_scores(semantic_scores)
    bm25_norm = _normalize_scores(bm25_scores)

    candidate_ids = set(semantic_norm) | set(bm25_norm)
    semantic_meta_by_id = {}
    for m in semantic_matches:
        cid = _get_chunk_id(m)
        if cid and cid not in semantic_meta_by_id:
            semantic_meta_by_id[cid] = _get_match_metadata(m)

    combined = []
    for cid in candidate_ids:
        s = semantic_norm.get(cid, 0.0)
        b = bm25_norm.get(cid, 0.0)
        score = semantic_weight * s + bm25_weight * b
        item = local_by_id.get(cid, {})
        metadata = item.get("metadata", {})
        text = item.get("text", "")

        if not text:
            pinecone_meta = semantic_meta_by_id.get(cid, {})
            text = pinecone_meta.get("text", "")
            metadata = metadata or pinecone_meta

        if text:
            adjusted_score = score * _source_multiplier(metadata)
            combined.append((cid, adjusted_score, text, metadata, s, b))

    combined.sort(key=lambda x: x[1], reverse=True)
    selected = combined[:top_k]
    docs = [entry[2] for entry in selected]

    if return_debug:
        hits = [
            {
                "id": cid,
                "score": score,
                "semantic_norm": sem,
                "bm25_norm": lex,
                "topic": meta.get("topic", ""),
                "source": meta.get("source", ""),
                "text": text,
            }
            for cid, score, text, meta, sem, lex in selected
        ]
        return {
            "docs": docs,
            "hits": hits,
            "latency_sec": time.perf_counter() - start,
            "strategy": strategy,
        }

    return docs
