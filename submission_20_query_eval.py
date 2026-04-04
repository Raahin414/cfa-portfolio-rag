import json
from pathlib import Path
from statistics import mean

from faithfulness import faithfulness_score
from generate_answer import generate_answer
from relevance import relevance_score

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
            top_k=6,
            strategy="semantic",
            semantic_weight=0.5,
            bm25_weight=0.5,
            generation_model="meta-llama/Meta-Llama-3-8B-Instruct",
            temperature=0.0,
            max_tokens=300,
        )

        faith = faithfulness_score(result["answer"], result["contexts"])["score"]
        rel = float(relevance_score(query, result["answer"]))
        lat = float(result["latency"]["total_sec"])
        backend = result.get("generation_backend", "unknown")

        rows.append(
            {
                "id": i,
                "query": query,
                "answer": result["answer"],
                "faithfulness": float(faith),
                "relevance": rel,
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
            "strategy": "semantic",
            "semantic_weight": 0.5,
            "bm25_weight": 0.5,
            "top_k": 6,
            "generation_model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "temperature": 0.0,
            "max_tokens": 300,
        },
        "avg_faithfulness": float(mean([x["faithfulness"] for x in rows])),
        "avg_relevance": float(mean([x["relevance"] for x in rows])),
        "avg_latency_sec": float(mean([x["latency_sec"] for x in rows])),
        "success_rate": float(
            mean([1.0 if x["backend"] == "huggingface_inference_api" else 0.0 for x in rows])
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
