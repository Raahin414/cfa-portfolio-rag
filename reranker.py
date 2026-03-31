from sentence_transformers import CrossEncoder


reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query, docs):
    ranked = rerank_with_scores(query, docs)
    return [item["doc"] for item in ranked]


def rerank_with_scores(query, docs):
    if not docs:
        return []

    pairs = [(query, doc) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [{"doc": doc, "score": float(score)} for doc, score in ranked]
