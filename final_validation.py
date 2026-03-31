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


def parse_args():
    parser = argparse.ArgumentParser(description="Run final end-to-end validation on 15 queries.")
    parser.add_argument("--strategy", required=True, choices=["fixed", "recursive", "semantic"], help="Best chunking strategy")
    parser.add_argument("--semantic-weight", type=float, required=True, help="Best semantic weight")
    parser.add_argument("--bm25-weight", type=float, required=True, help="Best BM25 weight")
    parser.add_argument("--output", default="final_validation_report.json", help="Output JSON report")
    return parser.parse_args()


def main():
    args = parse_args()

    rows = []
    for q in DEFAULT_QUERIES:
        result = generate_answer(
            q,
            top_k=8,
            strategy=args.strategy,
            semantic_weight=args.semantic_weight,
            bm25_weight=args.bm25_weight,
        )

        faith = faithfulness_score(result["answer"], result["contexts"])["score"]
        rel = relevance_score(q, result["answer"])
        rows.append({
            "query": q,
            "faithfulness": faith,
            "relevance": float(rel),
            "latency_sec": result["latency"]["total_sec"],
            "answer_preview": result["answer"][:240],
        })

    avg_f = mean([r["faithfulness"] for r in rows])
    avg_r = mean([r["relevance"] for r in rows])
    avg_l = mean([r["latency_sec"] for r in rows])

    report = {
        "strategy": args.strategy,
        "weights": {"semantic": args.semantic_weight, "bm25": args.bm25_weight},
        "average_faithfulness": avg_f,
        "average_relevance": avg_r,
        "average_latency_sec": avg_l,
        "rows": rows,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Final validation completed.")
    print(f"Average faithfulness: {avg_f:.3f}")
    print(f"Average relevance: {avg_r:.3f}")
    print(f"Average latency: {avg_l:.3f}s")
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
