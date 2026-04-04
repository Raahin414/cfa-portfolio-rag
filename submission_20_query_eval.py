import json
import os
from pathlib import Path
from statistics import mean

from faithfulness import faithfulness_score
from generate_answer import generate_answer
from relevance import relevance_score

GENERATION_MODEL = os.getenv("HF_GENERATION_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
STRATEGY = "semantic"
SEMANTIC_WEIGHT = 0.3
BM25_WEIGHT = 0.7
USE_RERANKER = False
RERANK_TOP_K = 5
TOP_K = 6

QUERIES = [
    "What is the efficient frontier in portfolio management?",
    "Explain the role of constraints in portfolio construction.",
    "Define and explain diversification in portfolio context.",
    "What is the relationship between risk and return?",
    "Describe the Modern Portfolio Theory.",
    "How is the Sharpe ratio calculated and what does it measure?",
    "Explain rebalancing and its importance in portfolio management.",
    "What is the significance of correlation in portfolio diversification?",
    "Describe the role of the efficient frontier in optimal portfolio selection.",
    "What is beta and how is it used in portfolio management?",
    "What are the key components of an Investment Policy Statement?",
    "How do time horizons affect portfolio construction decisions?",
    "What role does liquidity play in portfolio constraints?",
    "Explain tax considerations in portfolio planning.",
    "How do regulatory requirements impact portfolio construction?",
    "Discuss the relationship between systematic and unsystematic risk.",
    "How do you handle concentration limits in portfolio optimization?",
    "Explain the difference between strategic and tactical asset allocation.",
    "What factors should be considered in evaluating portfolio performance?",
    "How can investors achieve diversification across asset classes?",
]


def main():
    rows = []
    for i, query in enumerate(QUERIES, start=1):
        result = generate_answer(
            query,
            top_k=TOP_K,
            strategy=STRATEGY,
            semantic_weight=SEMANTIC_WEIGHT,
            bm25_weight=BM25_WEIGHT,
            use_reranker=USE_RERANKER,
            rerank_top_k=RERANK_TOP_K,
            generation_model=GENERATION_MODEL,
            temperature=0.0,
            max_tokens=300,
        )

        faith_report = faithfulness_score(result["answer"], result["contexts"])
        faith = float(faith_report["score"])

        rel_report = relevance_score(query, result["answer"], return_details=True)
        rel = float(rel_report["average_score"])

        lat = float(result["latency"]["total_sec"])
        backend = result.get("generation_backend", "unknown")

        rows.append(
            {
                "id": i,
                "query": query,
                "answer": result["answer"],
                "faithfulness": float(faith),
                "faithfulness_report": faith_report,
                "relevance": rel,
                "relevance_report": rel_report,
                "retrieval_sec": float(result["latency"].get("retrieval_sec", 0.0)),
                "generation_sec": float(result["latency"].get("generation_sec", 0.0)),
                "latency_sec": lat,
                "backend": backend,
            }
        )

        print(
            f"[{i:02d}/20] backend={backend} faith={faith:.3f} rel={rel:.3f} lat={lat:.2f}s"
        )

    summary = {
        "num_queries": len(rows),
        "config": {
            "strategy": STRATEGY,
            "semantic_weight": SEMANTIC_WEIGHT,
            "bm25_weight": BM25_WEIGHT,
            "top_k": TOP_K,
            "use_reranker": USE_RERANKER,
            "rerank_top_k": RERANK_TOP_K,
            "generation_model": GENERATION_MODEL,
            "temperature": 0.0,
            "max_tokens": 300,
        },
        "avg_faithfulness": float(mean([x["faithfulness"] for x in rows])),
        "avg_relevance": float(mean([x["relevance"] for x in rows])),
        "avg_retrieval_sec": float(mean([x["retrieval_sec"] for x in rows])),
        "avg_generation_sec": float(mean([x["generation_sec"] for x in rows])),
        "avg_latency_sec": float(mean([x["latency_sec"] for x in rows])),
        "success_rate": float(
            mean([
                1.0 if x["backend"] in {"huggingface_inference_api", "local_llm_fallback"} else 0.0
                for x in rows
            ])
        ),
    }

    payload = {"summary": summary, "rows": rows}
    out_path = Path("submission_20_query_results.json")
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
