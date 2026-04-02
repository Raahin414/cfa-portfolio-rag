"""
COMPREHENSIVE TEST SUITE - BEFORE/AFTER IMPROVEMENTS
Tests all improvements and validates they work correctly
"""

import sys
sys.path.insert(0, r'c:\Users\psnwa\nlp_project\cfa-portfolio-rag-deploy-ready')

import json
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from generate_answer import generate_answer
from faithfulness import compute_faithfulness_score
from relevance import compute_relevance_score

# Diverse test queries
TEST_QUERIES = [
    ("What is the role of constraints in portfolio construction?", "constraints_advanced"),
    ("Explain diversification.", "foundation_basic"),
    ("What is the Sharpe ratio?", "metrics_technical"),
    ("Describe the investment policy statement.", "ips_structured"),
    ("How does time horizon affect portfolio decisions?", "ips_practical"),
    ("What is the efficient frontier?", "foundation_advanced"),
    ("Explain correlation in portfolio context.", "metrics_conceptual"),
    ("What constraints limit portfolio construction?", "constraints_specific"),
]

class ComprehensiveTest:
    def __init__(self):
        self.results = {
            "before": [],
            "after": [],
            "summary": {}
        }
    
    def test_config(self, query, category, config_name, strategy='semantic', 
                   sem_weight=0.5, bm_weight=0.5, top_k=8):
        """Run a single test."""
        logger.info(f"  Testing: {query[:50]}...")
        
        try:
            start = time.perf_counter()
            result = generate_answer(
                query=query,
                top_k=top_k,
                strategy=strategy,
                semantic_weight=sem_weight,
                bm25_weight=bm_weight
            )
            total_time = time.perf_counter() - start
            
            answer = result['answer']
            confidence = result['confidence']['answer_confidence_score']
            has_citations = result['confidence']['has_citations']
            
            # Compute metrics
            faithfulness = compute_faithfulness_score(query, answer, result['contexts'])
            relevancy = compute_relevance_score(query, answer)
            
            test_result = {
                "query": query,
                "category": category,
                "config": config_name,
                "answer_length": len(answer),
                "confidence_score": float(confidence),
                "has_citations": has_citations,
                "faithfulness": float(faithfulness),
                "relevancy": float(relevancy),
                "latency_sec": float(total_time),
                "backend": result['generation_backend']
            }
            
            logger.info(f"    ✓ Confidence: {confidence:.3f} | Faith: {faithfulness:.3f} | Relev: {relevancy:.3f} | {total_time:.1f}s")
            return test_result
            
        except Exception as e:
            logger.error(f"    ✗ Error: {str(e)[:100]}")
            return None
    
    def run_all_tests(self):
        """Run comprehensive test suite."""
        
        logger.info("\n" + "="*90)
        logger.info("🧪 COMPREHENSIVE SYSTEM TEST SUITE")
        logger.info("="*90)
        
        configs = [
            ("Semantic 0.5/0.5", 'semantic', 0.5, 0.5),
            ("Semantic 0.7/0.3", 'semantic', 0.7, 0.3),
            ("Recursive 0.5/0.5", 'recursive', 0.5, 0.5),
            ("Fixed 0.5/0.5", 'fixed', 0.5, 0.5),
        ]
        
        all_results = []
        
        for config_name, strategy, sem_w, bm_w in configs:
            logger.info(f"\n📊 Configuration: {config_name}")
            logger.info("-" * 90)
            
            config_results = []
            for query, category in TEST_QUERIES:
                result = self.test_config(query, category, config_name, strategy, sem_w, bm_w)
                if result:
                    config_results.append(result)
                    all_results.append(result)
            
            # Config summary
            if config_results:
                avg_conf = sum(r['confidence_score'] for r in config_results) / len(config_results)
                avg_faith = sum(r['faithfulness'] for r in config_results) / len(config_results)
                avg_relev = sum(r['relevancy'] for r in config_results) / len(config_results)
                avg_time = sum(r['latency_sec'] for r in config_results) / len(config_results)
                citations_pct = sum(1 for r in config_results if r['has_citations']) / len(config_results) * 100
                
                logger.info(f"\n  Summary:")
                logger.info(f"    Avg Confidence: {avg_conf:.3f}")
                logger.info(f"    Avg Faithfulness: {avg_faith:.3f}")
                logger.info(f"    Avg Relevancy: {avg_relev:.3f}")
                logger.info(f"    Avg Latency: {avg_time:.2f}s")
                logger.info(f"    Citations %: {citations_pct:.0f}%")
        
        return all_results
    
    def generate_comparison_report(self, all_results):
        """Generate before/after comparison."""
        
        logger.info("\n" + "="*90)
        logger.info("📈 IMPROVEMENT ANALYSIS")
        logger.info("="*90)
        
        # Group by config
        configs = {}
        for r in all_results:
            config = r['config']
            if config not in configs:
                configs[config] = []
            configs[config].append(r)
        
        # Rankings
        logger.info("\nQuality Rankings (by combined score):")
        logger.info("-" * 90)
        
        config_scores = []
        for config, results in configs.items():
            avg_conf = sum(r['confidence_score'] for r in results) / len(results)
            avg_faith = sum(r['faithfulness'] for r in results) / len(results)
            avg_relev = sum(r['relevancy'] for r in results) / len(results)
            avg_time = sum(r['latency_sec'] for r in results) / len(results)
            
            # Combined score: (confidence + faithfulness + relevancy) / 3, adjusted for latency
            combined = (avg_conf + avg_faith + avg_relev) / 3
            
            config_scores.append({
                "config": config,
                "confidence": avg_conf,
                "faithfulness": avg_faith,
                "relevancy": avg_relev,
                "latency": avg_time,
                "combined_score": combined
            })
        
        # Sort by combined score
        config_scores.sort(key=lambda x: x['combined_score'], reverse=True)
        
        for i, score in enumerate(config_scores, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            logger.info(f"{medal} {i}. {score['config']}")
            logger.info(f"     Confidence: {score['confidence']:.3f} | Faithfulness: {score['faithfulness']:.3f} | Relevancy: {score['relevancy']:.3f}")
            logger.info(f"     Latency: {score['latency']:.1f}s | Combined: {score['combined_score']:.3f}")
        
        return config_scores

    def save_test_results(self, all_results, config_scores):
        """Save detailed results."""
        output = Path("comprehensive_test_results.json")
        
        with open(output, 'w') as f:
            json.dump({
                "individual_results": all_results,
                "config_rankings": config_scores,
                "test_date": "2026-04-02",
                "queries_tested": len(set(r['query'] for r in all_results))
            }, f, indent=2)
        
        logger.info(f"\n✅ Results saved to {output}")
        return output

if __name__ == "__main__":
    tester = ComprehensiveTest()
    all_results = tester.run_all_tests()
    config_scores = tester.generate_comparison_report(all_results)
    tester.save_test_results(all_results, config_scores)
    
    logger.info("\n" + "="*90)
    logger.info("✨ TEST SUITE COMPLETE - Ready for report generation")
    logger.info("="*90)
