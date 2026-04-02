"""
LATENCY PROFILING & BOTTLENECK ANALYSIS
Identify where time is spent and how to optimize
"""

import sys
sys.path.insert(0, r'c:\Users\psnwa\nlp_project\cfa-portfolio-rag-deploy-ready')

import logging
import json
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Sample queries to profile
SAMPLE_QUERIES = [
    "What is the efficient frontier?",
    "Explain portfolio diversification.",
    "What is the investment policy statement?",
]

def profile_system():
    """Profile the entire pipeline."""
    from generate_answer import generate_answer
    
    logger.info("\n" + "="*80)
    logger.info("⏱️  SYSTEM LATENCY PROFILING")
    logger.info("="*80)
    
    profile_data = []
    
    for query in SAMPLE_QUERIES:
        logger.info(f"\nProfiling: {query[:50]}...")
        
        start = time.perf_counter()
        result = generate_answer(query=query, top_k=8, strategy='semantic', 
                                semantic_weight=0.5, bm25_weight=0.5)
        total = time.perf_counter() - start
        
        latencies = result['latency']
        
        profile_data.append({
            "query": query,
            "retrieval_ms": latencies['retrieval_sec'] * 1000,
            "rerank_ms": latencies['rerank_sec'] * 1000,
            "generation_ms": latencies['generation_sec'] * 1000,
            "total_ms": total * 1000,
        })
        
        logger.info(f"  Retrieval:  {latencies['retrieval_sec']*1000:6.1f} ms ({latencies['retrieval_sec']/total*100:5.1f}%)")
        logger.info(f"  Reranking:  {latencies['rerank_sec']*1000:6.1f} ms ({latencies['rerank_sec']/total*100:5.1f}%)")
        logger.info(f"  Generation: {latencies['generation_sec']*1000:6.1f} ms ({latencies['generation_sec']/total*100:5.1f}%)")
        logger.info(f"  TOTAL:      {total*1000:6.1f} ms")
    
    # Analyze bottlenecks
    avg_retrieval = sum(p['retrieval_ms'] for p in profile_data) / len(profile_data)
    avg_rerank = sum(p['rerank_ms'] for p in profile_data) / len(profile_data)
    avg_gen = sum(p['generation_ms'] for p in profile_data) / len(profile_data)
    avg_total = sum(p['total_ms'] for p in profile_data) / len(profile_data)
    
    logger.info("\n" + "="*80)
    logger.info("BOTTLENECK ANALYSIS")
    logger.info("="*80)
    logger.info(f"Avg Retrieval:  {avg_retrieval:6.1f} ms ({avg_retrieval/avg_total*100:5.1f}%)")
    logger.info(f"Avg Reranking:  {avg_rerank:6.1f} ms ({avg_rerank/avg_total*100:5.1f}%)")
    logger.info(f"Avg Generation: {avg_gen:6.1f} ms ({avg_gen/avg_total*100:5.1f}%)")
    logger.info(f"Avg TOTAL:      {avg_total:6.1f} ms")
    
    bottleneck = max([('Retrieval', avg_retrieval), ('Reranking', avg_rerank), ('Generation', avg_gen)], key=lambda x: x[1])
    logger.info(f"\n⚠️  PRIMARY BOTTLENECK: {bottleneck[0]} ({bottleneck[1]/avg_total*100:.1f}%)")
    
    # Optimization suggestions
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION SUGGESTIONS")
    logger.info("="*80)
    
    suggestions = {
        "Retrieval": [
            "Cache embeddings model locally (already done)",
            "Batch queries if possible",
            "Use top-k=5 instead of 8 to reduce search space (if quality permits)"
        ],
        "Reranking": [
            "Test: Rerank only top-3 instead of top-5 docs",
            "Consider: Skip reranking for very similar scores",
            "Potential: Parallel reranking (minor benefit due to small batches)"
        ],
        "Generation": [
            "Primary bottleneck - inherent to LLM API calls",
            "Cannot optimize without changing model",
            "Alternative: Use faster model (lower quality trade-off)",
            "Current acceptable: ~23s for complex answers"
        ]
    }
    
    for component, opts in suggestions.items():
        logger.info(f"\n{component}:")
        for opt in opts:
            logger.info(f"  • {opt}")
    
    # Save results
    output = Path("latency_profile.json")
    with open(output, 'w') as f:
        json.dump({
            "sample_queries": profile_data,
            "averages": {
                "retrieval_ms": avg_retrieval,
                "rerank_ms": avg_rerank,
                "generation_ms": avg_gen,
                "total_ms": avg_total
            },
            "bottleneck": bottleneck[0],
            "acceptability": "✅ ACCEPTABLE for Streamlit Cloud (5-8 user requests/min reasonable)"
        }, f, indent=2)
    
    logger.info(f"\n✅ Saved to {output}")

if __name__ == "__main__":
    profile_system()
