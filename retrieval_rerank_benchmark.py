import json
import time
from pathlib import Path
from statistics import mean

from hybrid_retrieval import hybrid_search
from reranker import rerank_with_scores
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


def context_relevance(query, docs):
    if not docs:
        return 0.0
    text = "\n\n".join(docs[:5])
    return float(relevance_score(query, text))


def run_config(strategy, sem_w, bm_w, top_k):
    rows = []
    for q in QUERIES:
        t0 = time.perf_counter()
        retr = hybrid_search(
            q,
            top_k=top_k,
            strategy=strategy,
            semantic_weight=sem_w,
            bm25_weight=bm_w,
            return_debug=True,
        )
        t_retr = time.perf_counter() - t0

        docs = retr["docs"]
        pre_rel = context_relevance(q, docs)

        t1 = time.perf_counter()
        reranked = rerank_with_scores(q, docs)
        t_rr = time.perf_counter() - t1

        post_docs = [x["doc"] for x in reranked]
        post_rel = context_relevance(q, post_docs)

        rows.append(
            {
                "query": q,
                "pre_relevance": pre_rel,
                "post_relevance": post_rel,
                "lift": post_rel - pre_rel,
                "retrieval_latency_sec": t_retr,
                "rerank_latency_sec": t_rr,
                "total_latency_sec": t_retr + t_rr,
            }
        )

    avg_pre = mean([r["pre_relevance"] for r in rows])
    avg_post = mean([r["post_relevance"] for r in rows])
    avg_lift = mean([r["lift"] for r in rows])
    avg_total_lat = mean([r["total_latency_sec"] for r in rows])

    # Objective emphasizes retrieval quality + rerank gain with mild latency penalty.
    objective = 0.65 * avg_post + 0.25 * avg_lift + 0.10 * (1.0 / (1.0 + avg_total_lat))

    return {
        "config": {
            "strategy": strategy,
            "semantic_weight": sem_w,
            "bm25_weight": bm_w,
            "top_k": top_k,
        },
        "metrics": {
            "avg_pre_relevance": avg_pre,
            "avg_post_relevance": avg_post,
            "avg_rerank_lift": avg_lift,
            "avg_total_latency_sec": avg_total_lat,
            "objective": objective,
        },
        "rows": rows,
    }


def main():
    all_results = []
    configs = []
    for strategy in ["fixed", "recursive", "semantic"]:
        for sem_w, bm_w in [(0.5, 0.5), (0.6, 0.4), (0.4, 0.6)]:
            for top_k in [6, 8, 10]:
                configs.append((strategy, sem_w, bm_w, top_k))

    for i, (strategy, sem_w, bm_w, top_k) in enumerate(configs, start=1):
        print(f"[{i}/{len(configs)}] strategy={strategy} weights={sem_w}/{bm_w} top_k={top_k}")
        all_results.append(run_config(strategy, sem_w, bm_w, top_k))

    ranked = sorted(all_results, key=lambda x: x["metrics"]["objective"], reverse=True)

    payload = {
        "summary": {
            "num_queries": len(QUERIES),
            "num_configs": len(configs),
            "best": ranked[0],
            "top5": ranked[:5],
        },
        "all_results": all_results,
    }

    out_json = Path("retrieval_rerank_benchmark_results.json")
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out_md = Path("RETRIEVAL_RERANK_REPORT.md")
    lines = []
    lines.append("# Retrieval + Reranking Benchmark Report")
    lines.append("")
    lines.append(f"- Queries: {len(QUERIES)}")
    lines.append(f"- Configurations: {len(configs)}")
    lines.append("")
    b = ranked[0]
    c = b["config"]
    m = b["metrics"]
    lines.append("## Best Configuration")
    lines.append(f"- Chunking: {c['strategy']}")
    lines.append(f"- Weights: semantic={c['semantic_weight']}, bm25={c['bm25_weight']}")
    lines.append(f"- top_k: {c['top_k']}")
    lines.append("")
    lines.append("## Best Metrics")
    lines.append(f"- Avg pre-rerank relevance: {m['avg_pre_relevance']:.3f}")
    lines.append(f"- Avg post-rerank relevance: {m['avg_post_relevance']:.3f}")
    lines.append(f"- Avg rerank lift: {m['avg_rerank_lift']:.3f}")
    lines.append(f"- Avg total latency (s): {m['avg_total_latency_sec']:.3f}")
    lines.append(f"- Objective: {m['objective']:.3f}")
    lines.append("")
    lines.append("## Top 5")
    lines.append("| Rank | Strategy | Weights | top_k | PostRel | Lift | Latency(s) | Objective |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|")
    for i, item in enumerate(ranked[:5], start=1):
        c = item["config"]
        m = item["metrics"]
        lines.append(
            f"| {i} | {c['strategy']} | {c['semantic_weight']}/{c['bm25_weight']} | {c['top_k']} | {m['avg_post_relevance']:.3f} | {m['avg_rerank_lift']:.3f} | {m['avg_total_latency_sec']:.3f} | {m['objective']:.3f} |"
        )

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print("\nBest config:")
    print(json.dumps(b["config"], indent=2))
    print(json.dumps(b["metrics"], indent=2))
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
