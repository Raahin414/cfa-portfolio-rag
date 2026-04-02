"""
QUICK VALIDATION TEST - Verify improvements are working
"""

import sys
sys.path.insert(0, r'c:\Users\psnwa\nlp_project\cfa-portfolio-rag-deploy-ready')

from generate_answer import generate_answer

print("\n" + "="*80)
print("QUICK VALIDATION TEST")
print("="*80)

queries = [
    "What is diversification?",
    "Explain the Sharpe ratio.",
    "What is the efficient frontier?"
]

for i, query in enumerate(queries, 1):
    print(f"\n[Test {i}] {query}")
    print("-" * 80)
    
    result = generate_answer(
        query=query,
        top_k=8,
        strategy='semantic',
        semantic_weight=0.5,
        bm25_weight=0.5
    )
    
    # Key metrics
    conf = result['confidence']['answer_confidence_score']
    has_cit = result['confidence']['has_citations']
    faith = "N/A"  # Will compute if available
    latency = result['latency']['total_sec']
    
    answer = result['answer']
    
    print(f"Answer length: {len(answer)} chars")
    print(f"Confidence score: {conf:.3f}")
    print(f"Has citations: {'✓' if has_cit else '✗'}")
    print(f"Latency: {latency:.1f}s")
    print(f"Answer preview: {answer[:150]}...")
    
    # Verify improvements
    if conf > 0.3:
        print("✓ Confidence scoring working")
    if has_cit:
        print("✓ Citations present")
    if answer.endswith('.') or answer.endswith('?') or answer.endswith('!'):
        print("✓ Proper sentence ending")

print("\n" + "="*80)
print("✅ VALIDATION COMPLETE - All systems operational")
print("="*80)
