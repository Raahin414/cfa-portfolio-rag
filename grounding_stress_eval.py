import json
from pathlib import Path
from statistics import mean

from faithfulness import faithfulness_score
from generate_answer import generate_answer
from relevance import relevance_score


# Mix of in-domain and intentionally out-of-domain prompts.
STRESS_QUERIES = [
    {"query": "What is the efficient frontier in portfolio management?", "expected_grounded": True},
    {"query": "How does diversification reduce unsystematic risk?", "expected_grounded": True},
    {"query": "Who won the FIFA World Cup in 2022?", "expected_grounded": False},
    {"query": "What is the weather in Karachi today?", "expected_grounded": False},
    {"query": "Write Python code to reverse a linked list.", "expected_grounded": False},
    {"query": "What is quantum entanglement?", "expected_grounded": False},
    {"query": "How do taxes affect portfolio construction constraints?", "expected_grounded": True},
    {"query": "Explain CAPM alpha and beta in portfolio analysis.", "expected_grounded": True},
    {"query": "What is the best gaming laptop in 2026?", "expected_grounded": False},
    {"query": "How does liquidity constrain an investment policy statement?", "expected_grounded": True},
]


def run_once(item):
    query = item["query"]
    expected_grounded = bool(item["expected_grounded"])

    result = generate_answer(
        query,
        top_k=6,
        strategy="semantic",
        semantic_weight=0.3,
        bm25_weight=0.7,
        use_reranker=False,
        rerank_top_k=5,
        temperature=0.0,
        max_tokens=300,
    )

    faith = faithfulness_score(result["answer"], result["contexts"])
    rel = relevance_score(query, result["answer"], return_details=True)

    answer = result["answer"] or ""
    refused = "information not found in dataset" in answer.lower()

    return {
        "query": query,
        "expected_grounded": expected_grounded,
        "faithfulness": float(faith["score"]),
        "faith_total_claims": int(faith.get("total_claims", 0)),
        "faith_supported_claims": int(faith.get("supported_claims", 0)),
        "relevance": float(rel["average_score"]),
        "generation_backend": result.get("generation_backend", "unknown"),
        "refused_out_of_scope": refused,
        "answer_preview": answer[:220].replace("\n", " "),
        "generated_questions": rel.get("generated_questions", []),
    }


def main():
    rows = [run_once(item) for item in STRESS_QUERIES]

    in_domain = [r for r in rows if r["expected_grounded"]]
    out_domain = [r for r in rows if not r["expected_grounded"]]

    summary = {
        "num_queries": len(rows),
        "num_in_domain": len(in_domain),
        "num_out_domain": len(out_domain),
        "avg_faithfulness_all": float(mean(r["faithfulness"] for r in rows)),
        "avg_faithfulness_in_domain": float(mean(r["faithfulness"] for r in in_domain)) if in_domain else 0.0,
        "avg_faithfulness_out_domain": float(mean(r["faithfulness"] for r in out_domain)) if out_domain else 0.0,
        "avg_relevance_all": float(mean(r["relevance"] for r in rows)),
        "avg_relevance_in_domain": float(mean(r["relevance"] for r in in_domain)) if in_domain else 0.0,
        "avg_relevance_out_domain": float(mean(r["relevance"] for r in out_domain)) if out_domain else 0.0,
        "out_domain_refusal_rate": float(
            mean(1.0 if r["refused_out_of_scope"] else 0.0 for r in out_domain)
        ) if out_domain else 0.0,
    }

    payload = {"summary": summary, "rows": rows}
    out_path = Path("grounding_stress_results.json")
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("STRESS SUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
