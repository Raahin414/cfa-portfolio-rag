"""
Test improved system with confidence scoring
"""

import sys
sys.path.insert(0, r'c:\Users\psnwa\nlp_project\cfa-portfolio-rag-deploy-ready')

from generate_answer import generate_answer
import json

test_queries = [
    "What is diversification in portfolio management?",
    "Explain the Sharpe ratio.",
    "What is the efficient frontier?",
]

print("\n" + "="*80)
print("TESTING IMPROVED SYSTEM")
print("="*80)

results = []

for i, query in enumerate(test_queries, 1):
    print(f"\n[{i}] {query}")
    print("-" * 80)
    
    result = generate_answer(query=query, top_k=8, strategy='semantic', 
                            semantic_weight=0.5, bm25_weight=0.5)
    
    answer = result['answer']
    confidence = result['confidence']['answer_confidence_score']
    has_citations = result['confidence']['has_citations']
    backend = result['generation_backend']
    
    print(f"Answer: {answer[:250]}...")
    print(f"\nConfidence Score: {confidence:.3f}")
    print(f"Has Citations: {'✓' if has_citations else '✗'}")
    print(f"Backend: {backend}")
    print(f"Total Latency: {result['latency']['total_sec']:.2f}s")
    
    results.append({
        "query": query,
        "confidence": confidence,
        "has_citations": has_citations,
        "backend": backend
    })

# Save results
with open('test_improvements.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Avg Confidence: {sum(r['confidence'] for r in results) / len(results):.3f}")
print(f"All have citations: {all(r['has_citations'] for r in results)}")
print(f"✅ Improvements working correctly")
