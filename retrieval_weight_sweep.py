import json
from pathlib import Path
from statistics import mean

from faithfulness import faithfulness_score
from generate_answer import generate_answer
from relevance import relevance_score
from submission_20_query_eval import QUERIES


SEMANTIC_WEIGHTS = [0.3, 0.5, 0.7, 0.8]
TOP_K = 6
RERANK_TOP_K = 5
STRATEGY = "semantic"


def run_config(semantic_weight, use_reranker):
    bm25_weight = round(1.0 - semantic_weight, 2)
    rows = []
    for query in QUERIES:
        result = generate_answer(
            query,
            top_k=TOP_K,
            strategy=STRATEGY,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight,
            use_reranker=use_reranker,
            rerank_top_k=RERANK_TOP_K,
            temperature=0.0,
            max_tokens=280,
        )
        faith = faithfulness_score(result["answer"], result["contexts"])["score"]
        rel = relevance_score(query, result["answer"])
        lat = result["latency"]["total_sec"]
        rows.append(
            {
                "query": query,
                "faithfulness": float(faith),
                "relevance": float(rel),
                "latency_sec": float(lat),
                "backend": result.get("generation_backend", "unknown"),
            }
        )

    avg_faith = float(mean(r["faithfulness"] for r in rows))
    avg_rel = float(mean(r["relevance"] for r in rows))
    avg_lat = float(mean(r["latency_sec"] for r in rows))

    return {
        "config": {
            "strategy": STRATEGY,
            "semantic_weight": semantic_weight,
            "bm25_weight": bm25_weight,
            "top_k": TOP_K,
            "use_reranker": bool(use_reranker),
            "rerank_top_k": RERANK_TOP_K,
        },
        "metrics": {
            "avg_faithfulness": avg_faith,
            "avg_relevance": avg_rel,
            "avg_latency_sec": avg_lat,
            "objective": avg_faith + avg_rel,
        },
        "rows": rows,
    }


def main():
    all_results = []
    for sem in SEMANTIC_WEIGHTS:
        for rerank_flag in (False, True):
            print(
                f"Running config semantic={sem:.1f} bm25={1.0-sem:.1f} reranker={rerank_flag}..."
            )
            result = run_config(sem, rerank_flag)
            all_results.append(result)
            m = result["metrics"]
            print(
                f" -> faith={m['avg_faithfulness']:.3f} rel={m['avg_relevance']:.3f} lat={m['avg_latency_sec']:.2f} obj={m['objective']:.3f}"
            )

    all_results.sort(key=lambda x: x["metrics"]["objective"], reverse=True)
    best = all_results[0]

    payload = {
        "best": best,
        "all_results": all_results,
    }

    out_path = Path("retrieval_weight_sweep_results.json")
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Best config:")
    print(json.dumps(best["config"], indent=2))
    print(json.dumps(best["metrics"], indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
