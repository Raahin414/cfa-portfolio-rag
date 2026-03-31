import argparse
import json
from statistics import mean

from faithfulness import faithfulness_score
from generate_answer import generate_answer
from relevance import relevance_score


DEFAULT_QUERIES = [
    "What is diversification in portfolio management?",
    "What is asset allocation?",
    "What is portfolio risk?",
    "What is modern portfolio theory?",
    "What is the efficient frontier?",
    "How does diversification reduce risk?",
    "What is the Sharpe ratio?",
    "What is beta in portfolio management?",
    "What is passive vs active investing?",
    "What is portfolio rebalancing?",
    "How does CAPM relate to expected return?",
    "What is systematic vs unsystematic risk?",
    "How do fixed income securities affect portfolio risk?",
    "How do alternative investments improve diversification?",
    "What factors influence optimal asset allocation?",
]

WEIGHT_CONFIGS = [
    {"semantic": 0.7, "bm25": 0.3},
    {"semantic": 0.6, "bm25": 0.4},
    {"semantic": 0.5, "bm25": 0.5},
]


def evaluate_config(strategy, cfg, queries):
    rows = []
    for q in queries:
        result = generate_answer(
            q,
            top_k=8,
            strategy=strategy,
            semantic_weight=cfg["semantic"],
            bm25_weight=cfg["bm25"],
        )
        faith = faithfulness_score(result["answer"], result["contexts"])["score"]
        rel = relevance_score(q, result["answer"])
        lat = result["latency"]["total_sec"]
        rows.append({
            "query": q,
            "faithfulness": faith,
            "relevance": float(rel),
            "latency_sec": lat,
        })

    return {
        "strategy": strategy,
        "weights": cfg,
        "avg_faithfulness": mean([r["faithfulness"] for r in rows]),
        "avg_relevance": mean([r["relevance"] for r in rows]),
        "avg_latency_sec": mean([r["latency_sec"] for r in rows]),
        "rows": rows,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize hybrid retrieval weights.")
    parser.add_argument("--strategy", required=True, choices=["fixed", "recursive", "semantic"], help="Chunking strategy")
    parser.add_argument("--output", default="retrieval_optimization_results.json", help="Output JSON path")
    return parser.parse_args()


def main():
    args = parse_args()
    results = []

    for cfg in WEIGHT_CONFIGS:
        print(f"Testing weights semantic={cfg['semantic']} bm25={cfg['bm25']}...")
        results.append(evaluate_config(args.strategy, cfg, DEFAULT_QUERIES))

    print("\nWeights\tFaithfulness\tRelevance\tLatency")
    for r in results:
        w = r["weights"]
        print(
            f"{w['semantic']:.1f}/{w['bm25']:.1f}\t{r['avg_faithfulness']:.3f}\t\t"
            f"{r['avg_relevance']:.3f}\t\t{r['avg_latency_sec']:.3f}s"
        )

    best = max(results, key=lambda x: (x["avg_faithfulness"] + x["avg_relevance"], -x["avg_latency_sec"]))
    print(f"\nBest weights: semantic={best['weights']['semantic']} bm25={best['weights']['bm25']}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"strategy": args.strategy, "results": results, "best": best["weights"]}, f, indent=2)

    print(f"Saved optimization report to {args.output}")


if __name__ == "__main__":
    main()
