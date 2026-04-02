"""
Quick Evaluation - Sample 5 queries with detailed analysis for report
"""

import sys
sys.path.insert(0, r'c:\Users\psnwa\nlp_project\cfa-portfolio-rag-deploy-ready')

from generate_answer import generate_answer
from faithfulness import compute_faithfulness_score
from relevance import compute_relevance_score
import json
import time

# Sample 5 diverse queries for report examples
SAMPLE_QUERIES = [
    ("What is the role of constraints (liquidity, turnover, concentration limits) in real-world efficient portfolio construction?", "constraints"),
    ("Explain the Sharpe ratio and its significance in portfolio evaluation.", "metrics"),
    ("What is diversification and how does it reduce portfolio risk?", "foundation"),
    ("Describe the key components of an Investment Policy Statement.", "ips"),
    ("How do you balance strategic asset allocation with tactical adjustments?", "advanced"),
]

print("="*90)
print("CFA PORTFOLIO RAG - QUICK EVALUATION (5 QUERIES)")
print("="*90)
print()

evaluation_data = []

for i, (query, category) in enumerate(SAMPLE_QUERIES, 1):
    print(f"\n[Query {i}/{len(SAMPLE_QUERIES)}] {category.upper()}")
    print(f"Q: {query}")
    print("-" * 90)
    
    start = time.perf_counter()
    
    result = generate_answer(
        query=query,
        top_k=8,
        strategy='semantic',
        semantic_weight=0.5,
        bm25_weight=0.5
    )
    
    total_time = time.perf_counter() - start
    answer = result['answer']
    contexts = result['contexts']
    
    # Compute metrics
    faithfulness = compute_faithfulness_score(query, answer, contexts)
    relevancy = compute_relevance_score(query, answer)
    
    print(f"\nA: {answer[:300]}...")
    print(f"\nMetrics:")
    print(f"  Faithfulness Score: {faithfulness:.3f}")
    print(f"  Relevancy Score: {relevancy:.3f}")
    print(f"  Total Latency: {total_time:.2f}s")
    print(f"  Contexts Retrieved: {len(contexts)}")
    
    evaluation_data.append({
        "query_id": i,
        "category": category,
        "query": query,
        "answer_preview": answer[:300],
        "full_answer": answer,
        "faithfulness": float(faithfulness),
        "relevancy": float(relevancy),
        "latency_seconds": float(total_time),
        "num_contexts": len(contexts)
    })

# Save for report
with open('c:\\Users\\psnwa\\nlp_project\\cfa-portfolio-rag-deploy-ready\\quick_eval_results.json', 'w') as f:
    json.dump(evaluation_data, f, indent=2)

print("\n" + "="*90)
print("SUMMARY STATISTICS")
print("="*90)
faithfulness_scores = [e['faithfulness'] for e in evaluation_data]
relevancy_scores = [e['relevancy'] for e in evaluation_data]
latencies = [e['latency_seconds'] for e in evaluation_data]

print(f"Avg Faithfulness: {sum(faithfulness_scores)/len(faithfulness_scores):.3f}")
print(f"Avg Relevancy: {sum(relevancy_scores)/len(relevancy_scores):.3f}")
print(f"Avg Latency: {sum(latencies)/len(latencies):.2f}s")
print(f"\nResults saved to quick_eval_results.json")
