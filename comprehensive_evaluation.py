"""
Comprehensive Evaluation Suite for CFA Portfolio RAG System
Implements: Faithfulness, Relevancy, Latency Profiling, Ablation Study
"""

import sys
import json
import time
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, r'c:\Users\psnwa\nlp_project\cfa-portfolio-rag-deploy-ready')

from generate_answer import generate_answer
from hybrid_retrieval import hybrid_search
from relevance import compute_relevance_score
from faithfulness import compute_faithfulness_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 20 Diverse Test Queries covering different CFA portfolio topics
TEST_QUERIES = [
    # Foundation concepts (5)
    ("What is the efficient frontier in portfolio management?", "conceptual"),
    ("Explain the role of constraints in portfolio construction.", "conceptual"),
    ("Define and explain diversification in portfolio context.", "conceptual"),
    ("What is the relationship between risk and return?", "conceptual"),
    ("Describe the Modern Portfolio Theory.", "conceptual"),
    
    # Techniques & metrics (5)
    ("How is the Sharpe ratio calculated and what does it measure?", "technical"),
    ("Explain rebalancing and its importance in portfolio management.", "technical"),
    ("What is the significance of correlation in portfolio diversification?", "technical"),
    ("Describe the role of the efficient frontier in optimal portfolio selection.", "technical"),
    ("What is beta and how is it used in portfolio management?", "technical"),
    
    # Investment Policy Statement (5)
    ("What are the key components of an Investment Policy Statement?", "ips"),
    ("How do time horizons affect portfolio construction decisions?", "ips"),
    ("What role does liquidity play in portfolio constraints?", "ips"),
    ("Explain tax considerations in portfolio planning.", "ips"),
    ("How do regulatory requirements impact portfolio construction?", "ips"),
    
    # Advanced topics (5)
    ("Discuss the relationship between systematic and unsystematic risk.", "advanced"),
    ("How do you handle concentration limits in portfolio optimization?", "advanced"),
    ("Explain the difference between strategic and tactical asset allocation.", "advanced"),
    ("What factors should be considered in evaluating portfolio performance?", "advanced"),
    ("How can investors achieve diversification across asset classes?", "advanced"),
]

class ComprehensiveEvaluator:
    def __init__(self, output_dir="evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {
            "queries": [],
            "strategy_performance": defaultdict(list),
            "ablation_study": [],
            "latency_analysis": defaultdict(list)
        }

    def evaluate_query(self, query, top_k=8, strategy='semantic', semantic_weight=0.5, bm25_weight=0.5):
        """Evaluate a single query comprehensively."""
        logger.info(f"Evaluating: {query[:60]}...")
        
        # Measure total latency
        start_time = time.perf_counter()
        
        # Generate answer using pipeline
        result = generate_answer(
            query=query,
            top_k=top_k,
            strategy=strategy,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight
        )
        
        total_latency = time.perf_counter() - start_time
        
        answer = result['answer']
        contexts = result['contexts']
        
        # Compute Faithfulness Score
        faithfulness_score = compute_faithfulness_score(query, answer, contexts)
        
        # Compute Relevancy Score
        relevancy_score = compute_relevance_score(query, answer)
        
        # Extract latency metrics
        retrieval_time = result['latency'].get('retrieval_sec', 0)
        rerank_time = result['latency'].get('rerank_sec', 0)
        generation_time = result['latency'].get('generation_sec', 0)
        
        return {
            "query": query,
            "answer": answer,
            "contexts": contexts,
            "faithfulness": faithfulness_score,
            "relevancy": relevancy_score,
            "latency": {
                "retrieval": retrieval_time,
                "reranking": rerank_time,
                "generation": generation_time,
                "total": total_latency
            },
            "strategy": strategy,
            "weights": {"semantic": semantic_weight, "bm25": bm25_weight},
            "top_k": top_k
        }

    def run_comprehensive_evaluation(self, strategies=None):
        """Run evaluation across multiple strategies."""
        if strategies is None:
            strategies = [
                ("semantic", 0.5, 0.5),
                ("semantic", 0.7, 0.3),
                ("semantic", 0.3, 0.7),
                ("recursive", 0.5, 0.5),
                ("fixed", 0.5, 0.5),
            ]
        
        all_results = []
        
        for strategy, sem_weight, bm_weight in strategies:
            logger.info(f"\n{'='*80}")
            logger.info(f"Testing: {strategy} strategy (semantic: {sem_weight}, BM25: {bm_weight})")
            logger.info(f"{'='*80}\n")
            
            strategy_results = []
            strategy_metrics = {
                "strategy": strategy,
                "weights": (sem_weight, bm_weight),
                "faithfulness_scores": [],
                "relevancy_scores": [],
                "latencies": []
            }
            
            for query, category in TEST_QUERIES:
                try:
                    result = self.evaluate_query(
                        query,
                        strategy=strategy,
                        semantic_weight=sem_weight,
                        bm25_weight=bm_weight
                    )
                    strategy_results.append(result)
                    strategy_metrics["faithfulness_scores"].append(result["faithfulness"])
                    strategy_metrics["relevancy_scores"].append(result["relevancy"])
                    strategy_metrics["latencies"].append(result["latency"]["total"])
                    
                    logger.info(f"  Faithfulness: {result['faithfulness']:.3f} | Relevancy: {result['relevancy']:.3f}")
                    
                except Exception as e:
                    logger.error(f"  Failed: {str(e)[:100]}")
            
            # Compute aggregates
            if strategy_metrics["faithfulness_scores"]:
                strategy_metrics["avg_faithfulness"] = np.mean(strategy_metrics["faithfulness_scores"])
                strategy_metrics["std_faithfulness"] = np.std(strategy_metrics["faithfulness_scores"])
                strategy_metrics["avg_relevancy"] = np.mean(strategy_metrics["relevancy_scores"])
                strategy_metrics["std_relevancy"] = np.std(strategy_metrics["relevancy_scores"])
                strategy_metrics["avg_latency"] = np.mean(strategy_metrics["latencies"])
                
                logger.info(f"\n  Avg Faithfulness: {strategy_metrics['avg_faithfulness']:.3f} ± {strategy_metrics['std_faithfulness']:.3f}")
                logger.info(f"  Avg Relevancy: {strategy_metrics['avg_relevancy']:.3f} ± {strategy_metrics['std_relevancy']:.3f}")
                logger.info(f"  Avg Latency: {strategy_metrics['avg_latency']:.2f}s")
            
            self.results["strategy_performance"][f"{strategy}_{sem_weight}_{bm_weight}"] = strategy_metrics
            all_results.extend(strategy_results)
        
        return all_results

    def generate_report(self, results):
        """Generate evaluation report."""
        # Summary statistics
        faithfulness_scores = [r["faithfulness"] for r in results]
        relevancy_scores = [r["relevancy"] for r in results]
        latencies = [r["latency"]["total"] for r in results]
        
        report = {
            "total_queries_evaluated": len(results),
            "metrics": {
                "faithfulness": {
                    "mean": float(np.mean(faithfulness_scores)),
                    "std": float(np.std(faithfulness_scores)),
                    "min": float(np.min(faithfulness_scores)),
                    "max": float(np.max(faithfulness_scores))
                },
                "relevancy": {
                    "mean": float(np.mean(relevancy_scores)),
                    "std": float(np.std(relevancy_scores)),
                    "min": float(np.min(relevancy_scores)),
                    "max": float(np.max(relevancy_scores))
                },
                "latency": {
                    "mean_seconds": float(np.mean(latencies)),
                    "std": float(np.std(latencies)),
                    "min": float(np.min(latencies)),
                    "max": float(np.max(latencies))
                }
            },
            "sample_results": results[:3]  # Example results
        }
        
        return report

    def save_results(self, results):
        """Save detailed results to JSON."""
        output_file = self.output_dir / "evaluation_results.json"
        
        # Serialize results
        serializable_results = []
        for r in results:
            r_copy = r.copy()
            r_copy["contexts"] = [str(c)[:100] for c in r_copy["contexts"]]  # Truncate contexts
            serializable_results.append(r_copy)
        
        with open(output_file, 'w') as f:
            json.dump({
                "results": serializable_results,
                "summary": self.generate_report(results),
                "strategy_performance": {
                    k: {**v, 'faithfulness_scores': [float(x) for x in v['faithfulness_scores']],
                         'relevancy_scores': [float(x) for x in v['relevancy_scores']],
                         'latencies': [float(x) for x in v['latencies']]}
                    for k, v in self.results["strategy_performance"].items()
                }
            }, f, indent=2)
        
        logger.info(f"\nResults saved to {output_file}")
        return output_file


if __name__ == "__main__":
    print("\n" + "="*80)
    print("CFA Portfolio RAG - COMPREHENSIVE EVALUATION SUITE")
    print("="*80 + "\n")
    
    evaluator = ComprehensiveEvaluator()
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    # Save results
    results_file = evaluator.save_results(results)
    
    # Print summary
    summary = evaluator.generate_report(results)
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(json.dumps(summary, indent=2))
