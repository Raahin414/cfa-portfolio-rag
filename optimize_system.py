"""
INTERNAL OPTIMIZATION FRAMEWORK
Profile → Test → Improve → Validate → Update Code

Run this to systematically improve the RAG system BEFORE reporting.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
import logging

sys.path.insert(0, r'c:\Users\psnwa\nlp_project\cfa-portfolio-rag-deploy-ready')

from generate_answer import generate_answer
from faithfulness import compute_faithfulness_score
from relevance import compute_relevance_score

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Diverse test queries covering all categories
TEST_QUERIES = [
    "What is the efficient frontier?",
    "Explain portfolio diversification.",
    "What role do constraints play in portfolio construction?",
    "Define and explain the Sharpe ratio.",
    "What is the investment policy statement?",
    "How should time horizon affect portfolio decisions?",
    "Explain the relationship between risk and return.",
    "What is modern portfolio theory?",
    "How do you rebalance a portfolio effectively?",
    "Explain correlation and portfolio diversification.",
]

class SystemOptimizer:
    def __init__(self):
        self.results = defaultdict(list)
        self.best_config = None
        self.best_metrics = {"faithfulness": 0, "relevancy": 0, "latency": float('inf')}
        
    def evaluate_config(self, strategy, semantic_weight, bm25_weight, top_k=8, label=None):
        """Test a specific configuration."""
        config_label = label or f"{strategy}_{semantic_weight}_{bm25_weight}_k{top_k}"
        logger.info(f"\n📊 Testing: {config_label}")
        logger.info("=" * 80)
        
        metrics = {
            "faithfulness": [],
            "relevancy": [],
            "latency": [],
            "config": {
                "strategy": strategy,
                "semantic_weight": semantic_weight,
                "bm25_weight": bm25_weight,
                "top_k": top_k
            }
        }
        
        for i, query in enumerate(TEST_QUERIES, 1):
            try:
                start = time.perf_counter()
                result = generate_answer(
                    query=query,
                    top_k=top_k,
                    strategy=strategy,
                    semantic_weight=semantic_weight,
                    bm25_weight=bm25_weight
                )
                latency = time.perf_counter() - start
                
                answer = result['answer']
                contexts = result['contexts']
                
                faith = compute_faithfulness_score(query, answer, contexts)
                relevancy = compute_relevance_score(query, answer)
                
                metrics["faithfulness"].append(faith)
                metrics["relevancy"].append(relevancy)
                metrics["latency"].append(latency)
                
                logger.info(f"  [{i:2d}] Faithfulness: {faith:.3f} | Relevancy: {relevancy:.3f} | {latency:.1f}s")
                
            except Exception as e:
                logger.warning(f"  [{i:2d}] Error: {str(e)[:50]}")
        
        # Aggregate metrics
        if metrics["faithfulness"]:
            agg = {
                "faithfulness_mean": np.mean(metrics["faithfulness"]),
                "faithfulness_std": np.std(metrics["faithfulness"]),
                "relevancy_mean": np.mean(metrics["relevancy"]),
                "relevancy_std": np.std(metrics["relevancy"]),
                "latency_mean": np.mean(metrics["latency"]),
                "latency_max": np.max(metrics["latency"]),
            }
            
            logger.info(f"\n  ✓ Avg Faithfulness: {agg['faithfulness_mean']:.3f} ± {agg['faithfulness_std']:.3f}")
            logger.info(f"  ✓ Avg Relevancy: {agg['relevancy_mean']:.3f} ± {agg['relevancy_std']:.3f}")
            logger.info(f"  ✓ Avg Latency: {agg['latency_mean']:.2f}s (max: {agg['latency_max']:.2f}s)")
            
            metrics.update(agg)
            self.results[config_label] = metrics
            
            # Track best configuration
            score = (agg["faithfulness_mean"] + agg["relevancy_mean"]) / 2 - (agg["latency_mean"] / 100)
            if score > (self.best_metrics["faithfulness"] + self.best_metrics["relevancy"]) / 2:
                self.best_config = config_label
                self.best_metrics = agg
                logger.info(f"  🏆 NEW BEST CONFIG!")
        
        return metrics

    def run_optimization_suite(self):
        """Run comprehensive optimization tests."""
        logger.info("\n" + "="*80)
        logger.info("🚀 RETRIEVAL OPTIMIZATION")
        logger.info("="*80)
        
        # Test different semantic/BM25 weight combinations
        weight_combos = [
            (0.3, 0.7, "BM25-heavy"),
            (0.5, 0.5, "Balanced (current)"),
            (0.7, 0.3, "Semantic-heavy"),
            (0.6, 0.4, "Semantic-biased"),
        ]
        
        for sem, bm, label in weight_combos:
            self.evaluate_config("semantic", sem, bm, top_k=8, label=f"Weights_{label}")
        
        logger.info("\n" + "="*80)
        logger.info("🔍 TOP-K OPTIMIZATION")
        logger.info("="*80)
        
        # Test different top-k values
        for k in [5, 8, 10, 12]:
            self.evaluate_config("semantic", 0.5, 0.5, top_k=k, label=f"TopK_{k}")
        
        logger.info("\n" + "="*80)
        logger.info("📚 CHUNKING STRATEGY COMPARISON")
        logger.info("="*80)
        
        # Compare all chunking strategies
        for strategy in ["fixed", "recursive", "semantic"]:
            self.evaluate_config(strategy, 0.5, 0.5, top_k=8, label=f"Strategy_{strategy}")
        
        logger.info("\n" + "="*80)
        logger.info("🎯 HYBRID SEARCH ABLATION")
        logger.info("="*80)
        
        # Semantic only vs hybrid with different balances
        self.evaluate_config("semantic", 1.0, 0.0, top_k=8, label="Semantic_Only")
        self.evaluate_config("semantic", 0.0, 1.0, top_k=8, label="BM25_Only")
        
    def generate_optimization_report(self):
        """Generate internal optimization report."""
        logger.info("\n" + "="*80)
        logger.info("📈 OPTIMIZATION RESULTS SUMMARY")
        logger.info("="*80)
        
        # Sort by combined score
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: (x[1]["faithfulness_mean"] + x[1]["relevancy_mean"]) / 2,
            reverse=True
        )
        
        logger.info("\nTop 5 Configurations:")
        logger.info("-" * 80)
        for i, (config, metrics) in enumerate(sorted_results[:5], 1):
            logger.info(f"{i}. {config}")
            logger.info(f"   Faithfulness: {metrics['faithfulness_mean']:.3f} ± {metrics['faithfulness_std']:.3f}")
            logger.info(f"   Relevancy:    {metrics['relevancy_mean']:.3f} ± {metrics['relevancy_std']:.3f}")
            logger.info(f"   Latency:      {metrics['latency_mean']:.2f}s")
            logger.info()
        
        logger.info(f"\n🏆 BEST CONFIGURATION: {self.best_config}")
        logger.info(f"   Settings: {self.results[self.best_config]['config']}")
        logger.info(f"   Metrics: {self.best_metrics}")
        
        return sorted_results

    def save_results(self):
        """Save optimization results to JSON."""
        output = Path("optimization_results.json")
        
        # Serialize results
        serializable = {}
        for config, metrics in self.results.items():
            m = metrics.copy()
            m["faithfulness"] = [float(x) for x in m.get("faithfulness", [])]
            m["relevancy"] = [float(x) for x in m.get("relevancy", [])]
            m["latency"] = [float(x) for x in m.get("latency", [])]
            serializable[config] = m
        
        with open(output, 'w') as f:
            json.dump({
                "all_results": serializable,
                "best_config": self.best_config,
                "best_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v 
                                for k, v in self.best_metrics.items()}
            }, f, indent=2)
        
        logger.info(f"\n✅ Results saved to {output}")
        return output

if __name__ == "__main__":
    optimizer = SystemOptimizer()
    optimizer.run_optimization_suite()
    optimizer.generate_optimization_report()
    optimizer.save_results()
    
    logger.info("\n" + "="*80)
    logger.info("✨ OPTIMIZATION COMPLETE")
    logger.info("Next: Update generate_answer.py with best configuration")
    logger.info("="*80)
