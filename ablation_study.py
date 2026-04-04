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

STRATEGIES = ["fixed", "recursive", "semantic"]


def evaluate_strategy(strategy, queries, semantic_weight=0.7, bm25_weight=0.3, use_reranker=True):
    rows = []
    for q in queries:
        result = generate_answer(
            q,
            top_k=8,
            strategy=strategy,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight,
            use_reranker=use_reranker,
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
        "avg_faithfulness": mean([r["faithfulness"] for r in rows]),
        "avg_relevance": mean([r["relevance"] for r in rows]),
        "avg_latency_sec": mean([r["latency_sec"] for r in rows]),
        "rows": rows,
    }


def print_table(results):
    print("\nChunking\tFaithfulness\tRelevance\tLatency")
    for r in results:
        print(
            f"{r['strategy']}\t{r['avg_faithfulness']:.3f}\t\t"
            f"{r['avg_relevance']:.3f}\t\t{r['avg_latency_sec']:.3f}s"
        )


def choose_best(results):
    return max(results, key=lambda x: (x["avg_faithfulness"] + x["avg_relevance"], -x["avg_latency_sec"]))


def parse_args():
    parser = argparse.ArgumentParser(description="Run ablation across chunking strategies.")
    parser.add_argument("--output", default="ablation_results.json", help="Output JSON path")
    parser.add_argument("--semantic-weight", type=float, default=0.7, help="Semantic retrieval weight")
    parser.add_argument("--bm25-weight", type=float, default=0.3, help="BM25 retrieval weight")
    parser.add_argument("--disable-reranker", action="store_true", help="Disable reranker during ablation")
    return parser.parse_args()


def main():
    args = parse_args()

    all_results = []
    for strategy in STRATEGIES:
        print(f"Running ablation for strategy={strategy}...")
        all_results.append(
            evaluate_strategy(
                strategy,
                DEFAULT_QUERIES,
                semantic_weight=args.semantic_weight,
                bm25_weight=args.bm25_weight,
                use_reranker=not args.disable_reranker,
            )
        )

    print_table(all_results)
    best = choose_best(all_results)
    print(f"\nBest chunking strategy: {best['strategy']}")

    payload = {
        "queries": DEFAULT_QUERIES,
        "config": {
            "semantic_weight": args.semantic_weight,
            "bm25_weight": args.bm25_weight,
            "use_reranker": not args.disable_reranker,
        },
        "results": all_results,
        "best_strategy": best["strategy"],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved ablation report to {args.output}")


if __name__ == "__main__":
    main()
