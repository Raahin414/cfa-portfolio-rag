import json
import os
import time
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

from faithfulness import faithfulness_score
from generate_answer import generate_answer
from relevance import relevance_score


load_dotenv(r"c:\Users\psnwa\nlp_project\.env")


REP_QUERIES = [
    "What is diversification in portfolio management?",
    "Explain the Sharpe ratio.",
    "What is the efficient frontier?",
    "What role do constraints play in portfolio construction?",
    "How does correlation affect portfolio risk?",
    "What is the purpose of rebalancing?",
    "What are key components of an investment policy statement?",
    "Differentiate systematic vs unsystematic risk.",
]

FULL_QUERIES = [
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


def score_row(result, query):
    ans = result["answer"]
    ctx = result["contexts"]
    faith = faithfulness_score(ans, ctx)["score"]
    rel = float(relevance_score(query, ans))
    lat = float(result["latency"]["total_sec"])
    backend = result.get("generation_backend", "unknown")
    return {
        "faithfulness": float(faith),
        "relevance": rel,
        "latency_sec": lat,
        "backend": backend,
    }


def aggregate(rows):
    faith = [r["faithfulness"] for r in rows]
    rel = [r["relevance"] for r in rows]
    lat = [r["latency_sec"] for r in rows]
    success = [1 if r["backend"] == "huggingface_inference_api" else 0 for r in rows]
    return {
        "avg_faithfulness": float(mean(faith)) if faith else 0.0,
        "avg_relevance": float(mean(rel)) if rel else 0.0,
        "avg_latency_sec": float(mean(lat)) if lat else 0.0,
        "generation_success_rate": float(mean(success)) if success else 0.0,
    }


def objective(metrics):
    # Prioritize groundedness for assignment: faithfulness > relevance > latency.
    return (
        0.55 * metrics["avg_faithfulness"]
        + 0.35 * metrics["avg_relevance"]
        + 0.10 * (1.0 / (1.0 + metrics["avg_latency_sec"]))
    )


def eval_config(config, queries):
    rows = []
    for q in queries:
        try:
            r = generate_answer(
                q,
                top_k=config["top_k"],
                strategy=config["strategy"],
                semantic_weight=config["semantic_weight"],
                bm25_weight=config["bm25_weight"],
                generation_model=config["model"],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
            )
            row = score_row(r, q)
            row["query"] = q
            rows.append(row)
        except Exception as e:
            rows.append(
                {
                    "query": q,
                    "faithfulness": 0.0,
                    "relevance": 0.0,
                    "latency_sec": 999.0,
                    "backend": f"error:{type(e).__name__}",
                }
            )

    metrics = aggregate(rows)
    metrics["objective"] = float(objective(metrics))
    return rows, metrics


def phase_a_retrieval_sweep(base_model):
    retrieval_configs = []
    for strategy in ["fixed", "recursive", "semantic"]:
        for sem, bm in [(0.5, 0.5), (0.6, 0.4)]:
            for top_k in [6, 8]:
                retrieval_configs.append(
                    {
                        "phase": "A_retrieval",
                        "model": base_model,
                        "strategy": strategy,
                        "semantic_weight": sem,
                        "bm25_weight": bm,
                        "top_k": top_k,
                        "temperature": 0.0,
                        "max_tokens": 260,
                    }
                )

    out = []
    for i, cfg in enumerate(retrieval_configs, start=1):
        print(f"[A {i}/{len(retrieval_configs)}] {cfg}")
        rows, metrics = eval_config(cfg, REP_QUERIES)
        out.append({"config": cfg, "metrics": metrics, "rows": rows})

    ranked = sorted(out, key=lambda x: x["metrics"]["objective"], reverse=True)
    return out, ranked[:3]


def phase_b_generation_sweep(top_retrieval_configs):
    out = []
    candidates = []
    for base in top_retrieval_configs:
        rc = base["config"]
        for model in [
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
        ]:
            for temperature in [0.0, 0.1]:
                for max_tokens in [220, 300]:
                    cfg = {
                        **rc,
                        "phase": "B_generation",
                        "model": model,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    candidates.append(cfg)

    for i, cfg in enumerate(candidates, start=1):
        print(f"[B {i}/{len(candidates)}] {cfg}")
        rows, metrics = eval_config(cfg, REP_QUERIES)
        out.append({"config": cfg, "metrics": metrics, "rows": rows})

    ranked = sorted(out, key=lambda x: x["metrics"]["objective"], reverse=True)
    return out, ranked[:4]


def phase_c_final_eval(finalists):
    out = []
    for i, item in enumerate(finalists, start=1):
        cfg = item["config"]
        cfg = {**cfg, "phase": "C_final"}
        print(f"[C {i}/{len(finalists)}] {cfg}")
        rows, metrics = eval_config(cfg, FULL_QUERIES)
        out.append({"config": cfg, "metrics": metrics, "rows": rows})
    return sorted(out, key=lambda x: x["metrics"]["objective"], reverse=True)


def write_report(all_data, report_md_path):
    lines = []
    lines.append("# Assignment Benchmark Report (Extensive Pipeline Evaluation)")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Retrieval: hybrid semantic + BM25 weights, top-k, chunking strategy")
    lines.append("- Reranking: CrossEncoder included in normal pipeline")
    lines.append("- Generation: model, temperature, max_tokens")
    lines.append("- Metrics: faithfulness, relevance, latency, generation success rate")
    lines.append("")

    final = all_data["phase_c_ranked"]
    best = final[0]
    bcfg = best["config"]
    bmet = best["metrics"]

    lines.append("## Best Final Configuration")
    lines.append(f"- Model: {bcfg['model']}")
    lines.append(f"- Chunking strategy: {bcfg['strategy']}")
    lines.append(f"- Weights: semantic={bcfg['semantic_weight']}, bm25={bcfg['bm25_weight']}")
    lines.append(f"- top_k: {bcfg['top_k']}")
    lines.append(f"- temperature: {bcfg['temperature']}")
    lines.append(f"- max_tokens: {bcfg['max_tokens']}")
    lines.append("")
    lines.append("## Best Final Metrics (20-query set)")
    lines.append(f"- Avg faithfulness: {bmet['avg_faithfulness']:.3f}")
    lines.append(f"- Avg relevance: {bmet['avg_relevance']:.3f}")
    lines.append(f"- Avg latency (s): {bmet['avg_latency_sec']:.3f}")
    lines.append(f"- Generation success rate: {bmet['generation_success_rate']:.3f}")
    lines.append(f"- Objective score: {bmet['objective']:.3f}")
    lines.append("")

    lines.append("## Finalist Comparison")
    lines.append("| Rank | Model | Strategy | Weights | top_k | Temp | Tokens | Faith | Rel | Latency(s) | Success | Objective |")
    lines.append("|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, item in enumerate(final, start=1):
        c = item["config"]
        m = item["metrics"]
        lines.append(
            f"| {i} | {c['model']} | {c['strategy']} | {c['semantic_weight']}/{c['bm25_weight']} | {c['top_k']} | {c['temperature']} | {c['max_tokens']} | {m['avg_faithfulness']:.3f} | {m['avg_relevance']:.3f} | {m['avg_latency_sec']:.3f} | {m['generation_success_rate']:.3f} | {m['objective']:.3f} |"
        )

    lines.append("")
    lines.append("## Ablation Insights")
    lines.append("- Chunking strategy impact is reflected by holding generation settings fixed in phase A.")
    lines.append("- Retrieval weight impact is reflected by comparing 0.5/0.5 vs 0.6/0.4.")
    lines.append("- Generation model/temperature/token impact is reflected in phase B and validated in phase C.")
    lines.append("")
    lines.append("## Notes")
    lines.append("- This report is generated from actual runs against the deployed Pinecone index.")
    lines.append("- Full machine-readable results are in `extensive_benchmark_results.json`.")

    report_md_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    t0 = time.perf_counter()

    default_model = os.getenv("HF_GENERATION_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

    phase_a_all, phase_a_top3 = phase_a_retrieval_sweep(default_model)
    phase_b_all, phase_b_top4 = phase_b_generation_sweep(phase_a_top3)
    phase_c_ranked = phase_c_final_eval(phase_b_top4)

    all_data = {
        "run_started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_sec": time.perf_counter() - t0,
        "phase_a_all": phase_a_all,
        "phase_a_top3": phase_a_top3,
        "phase_b_all": phase_b_all,
        "phase_b_top4": phase_b_top4,
        "phase_c_ranked": phase_c_ranked,
    }

    out_json = Path("extensive_benchmark_results.json")
    out_md = Path("ASSIGNMENT_BENCHMARK_REPORT.md")

    out_json.write_text(json.dumps(all_data, indent=2), encoding="utf-8")
    write_report(all_data, out_md)

    best = phase_c_ranked[0]
    print("\n=== BEST CONFIG (FINAL) ===")
    print(json.dumps(best["config"], indent=2))
    print(json.dumps(best["metrics"], indent=2))
    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
