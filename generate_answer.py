import argparse
import os
import time

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from hybrid_retrieval import hybrid_search
from reranker import rerank_with_scores


DEFAULT_QUERY = "How to diversify a portfolio to reduce risk?"
DEFAULT_HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


load_dotenv()


def build_prompt(query, contexts):
    context_block = "\n\n".join([f"Context {i + 1}: {c}" for i, c in enumerate(contexts)])
    return (
        "You are a finance expert answering questions ONLY using provided context.\n\n"
        "IMPORTANT RULES:\n"
        "1. Answer ONLY using the provided context below\n"
        "2. If the answer is not in the context, respond exactly: 'Information not found in dataset'\n"
        "3. Do NOT add external knowledge or general finance theory\n"
        "4. Do NOT hallucinate or invent information\n"
        "5. Be concise - keep answer under 120 words\n"
        "6. Prefer direct phrasing from context (extractive style)\n"
        "7. Add inline citations like [Context 1], [Context 2]\n\n"
        f"Question: {query}\n\n"
        f"PROVIDED CONTEXT:\n{context_block}\n\n"
        "Answer (grounded only in provided context):"
    )


def hf_generate(prompt):
    hf_token = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    model_name = os.getenv("HF_GENERATION_MODEL", DEFAULT_HF_MODEL)
    provider = os.getenv("HF_PROVIDER")
    client = InferenceClient(model=model_name, token=hf_token, provider=provider) if provider else InferenceClient(model=model_name, token=hf_token)

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer ONLY from provided context. "
                        "If missing, reply exactly: 'Information not found in dataset'. "
                        "No external knowledge."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=180,
            temperature=0.0,
        )
        return completion.choices[0].message.content.strip()
    except Exception:
        pass

    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    generated = client.text_generation(
        prompt=formatted_prompt,
        max_new_tokens=180,
        temperature=0.0,
        do_sample=True,
    )
    if isinstance(generated, str):
        return generated.strip()
    return str(generated).strip()


def fallback_generate(query, contexts):
    if not contexts:
        return "I do not have enough retrieved context to answer this question."

    best = contexts[0].strip()
    return (
        "Based on the retrieved context, here is the most relevant information:\n\n"
        f"{best}\n\n"
        f"Question addressed: {query}"
    )


def generate_answer(query, top_k=8, strategy="recursive", semantic_weight=0.7, bm25_weight=0.3):
    start = time.perf_counter()
    retrieval = hybrid_search(
        query,
        top_k=top_k or 8,
        strategy=strategy,
        semantic_weight=semantic_weight,
        bm25_weight=bm25_weight,
        return_debug=True,
    )
    docs = retrieval["docs"]
    hits = retrieval.get("hits", [])
    hit_by_text = {h.get("text", ""): h for h in hits if h.get("text")}

    rerank_start = time.perf_counter()
    ranked_docs = rerank_with_scores(query, docs)
    rerank_latency = time.perf_counter() - rerank_start
    selected_ranked = ranked_docs[:5]
    selected = [item["doc"] for item in selected_ranked]

    sources = []
    rerank_scores = [item["score"] for item in selected_ranked]
    for item in selected_ranked:
        text = item["doc"]
        hit = hit_by_text.get(text, {})
        sources.append(
            {
                "source": hit.get("source", ""),
                "topic": hit.get("topic", ""),
                "retrieval_score": float(hit.get("score", 0.0)),
                "rerank_score": float(item["score"]),
                "preview": text[:260].replace("\n", " "),
            }
        )

    prompt = build_prompt(query, selected)

    generation_start = time.perf_counter()
    try:
        llm_answer = hf_generate(prompt)
    except Exception:
        llm_answer = None
    generation_latency = time.perf_counter() - generation_start
    answer = llm_answer if llm_answer else fallback_generate(query, selected)
    backend = "huggingface_inference_api" if llm_answer else "fallback_context_only"

    return {
        "query": query,
        "answer": answer,
        "contexts": selected,
        "latency": {
            "retrieval_sec": retrieval["latency_sec"],
            "rerank_sec": rerank_latency,
            "generation_sec": generation_latency,
            "total_sec": time.perf_counter() - start,
        },
        "strategy": strategy,
        "weights": {
            "semantic": semantic_weight,
            "bm25": bm25_weight,
        },
        "generation_backend": backend,
        "sources": sources,
        "confidence": {
            "rerank_mean": (sum(rerank_scores) / len(rerank_scores)) if rerank_scores else 0.0,
            "num_supporting_contexts": len(selected),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate an answer from retrieved and reranked chunks.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="User question")
    parser.add_argument("--top-k", type=int, default=8, help="Number of docs to retrieve before rerank")
    parser.add_argument("--strategy", default="recursive", choices=["fixed", "recursive", "semantic"], help="Chunking strategy namespace")
    parser.add_argument("--semantic-weight", type=float, default=0.7, help="Weight for semantic retrieval")
    parser.add_argument("--bm25-weight", type=float, default=0.3, help="Weight for BM25 retrieval")
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    result = generate_answer(
        args.query,
        top_k=args.top_k,
        strategy=args.strategy,
        semantic_weight=args.semantic_weight,
        bm25_weight=args.bm25_weight,
    )

    print("Question:")
    print(result["query"])
    print("\nAnswer:")
    print(result["answer"])
    print("\nTop Contexts Used:")
    for i, ctx in enumerate(result["contexts"], start=1):
        preview = ctx[:220].replace("\n", " ")
        print(f"{i}. {preview}...")
    print("\nLatency:")
    print(f"- retrieval_sec: {result['latency']['retrieval_sec']:.3f}")
    print(f"- rerank_sec: {result['latency']['rerank_sec']:.3f}")
    print(f"- generation_sec: {result['latency']['generation_sec']:.3f}")
    print(f"- total_sec: {result['latency']['total_sec']:.3f}")
    print(f"- generation_backend: {result['generation_backend']}")


if __name__ == "__main__":
    main()
