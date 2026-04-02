"""
VALIDATED OPTIMIZATION - Switch to fixed chunking based on actual experiments
This implementation is based on real test results showing:
- Fixed chunking: 0.8977 combined score (1.0 faithfulness)
- Semantic chunking: 0.8972 combined score (0.958 faithfulness)
- Improvement: +0.05% with better faithfulness

DO NOT COMMIT UNTIL VERIFIED ON LIVE QUERIES
"""

import sys
sys.path.insert(0, r'c:\Users\psnwa\nlp_project\cfa-portfolio-rag-deploy-ready')

from generate_answer import generate_answer
from faithfulness import faithfulness_score
from relevance import relevance_score
import time
import json

# Test queries to verify the improvement works
VALIDATION_QUERIES = [
    "What is diversification in portfolio management?",
    "Explain the Sharpe ratio and its importance.",
    "What role do constraints play in portfolio construction?",
    "Describe the efficient frontier.",
    "What are the benefits of portfolio rebalancing?",
]

def validate_fixed_chunking():
    """Verify that fixed chunking actually improves results in production."""
    
    print("=" * 80)
    print("✅ VALIDATING FIXED CHUNKING IMPROVEMENT")
    print("=" * 80)
    
    results = {
        "baseline_semantic": [],
        "improved_fixed": [],
        "comparison": {}
    }
    
    for query in VALIDATION_QUERIES:
        print(f"\n📝 Testing: {query[:50]}...")
        
        # Test current (semantic)
        start = time.perf_counter()
        baseline = generate_answer(query, strategy='semantic')
        baseline_time = time.perf_counter() - start
        
        baseline_faith = faithfulness_score(baseline['answer'], baseline['contexts'])
        baseline_relev = relevance_score(query, baseline['answer'])
        
        baseline_score = {
            "query": query,
            "faithfulness": baseline_faith['score'],
            "relevancy": baseline_relev,
            "latency": baseline_time,
            "answer_length": len(baseline['answer'])
        }
        results["baseline_semantic"].append(baseline_score)
        
        # Test improved (fixed)
        start = time.perf_counter()
        improved = generate_answer(query, strategy='fixed')
        improved_time = time.perf_counter() - start
        
        improved_faith = faithfulness_score(improved['answer'], improved['contexts'])
        improved_relev = relevance_score(query, improved['answer'])
        
        improved_score = {
            "query": query,
            "faithfulness": improved_faith['score'],
            "relevancy": improved_relev,
            "latency": improved_time,
            "answer_length": len(improved['answer'])
        }
        results["improved_fixed"].append(improved_score)
        
        # Compare
        faith_diff = improved_faith['score'] - baseline_faith['score']
        relev_diff = improved_relev - baseline_relev
        latency_diff = improved_time - baseline_time
        
        print(f"  Semantic (baseline):")
        print(f"    Faith: {baseline_faith['score']:.4f} | Relev: {baseline_relev:.4f} | Time: {baseline_time:.2f}s")
        print(f"  Fixed (improved):")
        print(f"    Faith: {improved_faith['score']:.4f} | Relev: {improved_relev:.4f} | Time: {improved_time:.2f}s")
        print(f"  Diff:")
        print(f"    Δ Faith: {faith_diff:+.4f} | Δ Relev: {relev_diff:+.4f} | Δ Time: {latency_diff:+.2f}s")
    
    # Summary statistics
    import numpy as np
    
    baseline_faiths = [r['faithfulness'] for r in results["baseline_semantic"]]
    improved_faiths = [r['faithfulness'] for r in results["improved_fixed"]]
    
    baseline_relevs = [r['relevancy'] for r in results["baseline_semantic"]]
    improved_relevs = [r['relevancy'] for r in results["improved_fixed"]]
    
    results["comparison"] = {
        "baseline_avg_faithfulness": float(np.mean(baseline_faiths)),
        "improved_avg_faithfulness": float(np.mean(improved_faiths)),
        "faith_improvement": float(np.mean(improved_faiths) - np.mean(baseline_faiths)),
        
        "baseline_avg_relevancy": float(np.mean(baseline_relevs)),
        "improved_avg_relevancy": float(np.mean(improved_relevs)),
        "relev_improvement": float(np.mean(improved_relevs) - np.mean(baseline_relevs)),
        
        "baseline_avg_latency": float(np.mean([r['latency'] for r in results["baseline_semantic"]])),
        "improved_avg_latency": float(np.mean([r['latency'] for r in results["improved_fixed"]])),
    }
    
    print("\n" + "=" * 80)
    print("📊 PRODUCTION VALIDATION RESULTS")
    print("=" * 80)
    print(f"\nBaseline (SEMANTIC): Faith={results['comparison']['baseline_avg_faithfulness']:.4f}, "
          f"Relev={results['comparison']['baseline_avg_relevancy']:.4f}")
    print(f"Improved  (FIXED):   Faith={results['comparison']['improved_avg_faithfulness']:.4f}, "
          f"Relev={results['comparison']['improved_avg_relevancy']:.4f}")
    
    print(f"\nFaithfulness Improvement: {results['comparison']['faith_improvement']:+.4f} "
          f"({results['comparison']['faith_improvement']*100:+.2f}%)")
    print(f"Relevancy Improvement:    {results['comparison']['relev_improvement']:+.4f} "
          f"({results['comparison']['relev_improvement']*100:+.2f}%)")
    
    # Save results
    with open("fixed_chunking_validation.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Validation results saved to: fixed_chunking_validation.json")
    
    # Determine if improvement is real
    if results['comparison']['faith_improvement'] >= 0:
        print(f"\n✅ VALIDATION PASSED: Fixed chunking shows improvement or equivalent performance")
        print(f"   → Ready to implement and commit")
        return True
    else:
        print(f"\n⚠️  VALIDATION WARNING: Fixed chunking shows slight degradation")
        print(f"   → Keep current semantic strategy")
        return False

if __name__ == "__main__":
    success = validate_fixed_chunking()
    exit(0 if success else 1)
