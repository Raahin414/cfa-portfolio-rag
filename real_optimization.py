"""
REAL SYSTEM OPTIMIZATION - Actually test and find best configuration
Do NOT commit results until validated
"""

import sys
sys.path.insert(0, r'c:\Users\psnwa\nlp_project\cfa-portfolio-rag-deploy-ready')

import os
import json
import time
import numpy as np
import logging
from pathlib import Path
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

from faithfulness import faithfulness_score
from relevance import relevance_score
from hybrid_retrieval import hybrid_search
from reranker import rerank_with_scores

# 10 test queries to systematically evaluate
TEST_QUERIES = [
    "What is diversification in portfolio management?",
    "Explain the Sharpe ratio.",
    "What is the efficient frontier?",
    "What role do constraints play in portfolio construction?",
    "Describe the investment policy statement.",
    "How does correlation affect portfolio risk?",
    "What is the purpose of rebalancing?",
    "Explain systematic vs unsystematic risk.",
    "What are the benefits of asset allocation?",
    "How do you measure portfolio performance?",
]

class RealOptimizer:
    def __init__(self):
        self.results = {
            "llm_comparison": {},
            "parameter_tuning": {},
            "weight_optimization": {},
            "chunk_strategy_comparison": {}
        }
        self.best_config = None
    
    def test_llm_model(self, model_name, test_queries_subset=None):
        """Test a specific LLM model."""
        if test_queries_subset is None:
            test_queries_subset = TEST_QUERIES[:3]  # Use 3 queries for speed
        
        hf_token = os.getenv("HF_API_KEY")
        provider = os.getenv("HF_PROVIDER")
        
        logger.info(f"\n📊 Testing LLM: {model_name}")
        logger.info("-" * 80)
        
        metrics = []
        failed = 0
        
        for query in test_queries_subset:
            try:
                # Do retrieval first
                retrieval = hybrid_search(query, top_k=8, strategy='semantic',
                                         semantic_weight=0.5, bm25_weight=0.5)
                docs = retrieval["docs"]
                ranked_docs = rerank_with_scores(query, docs)[:5]
                contexts = [item["doc"] for item in ranked_docs]
                
                # Test LLM
                start = time.perf_counter()
                
                client = InferenceClient(model=model_name, token=hf_token, provider=provider)
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a finance expert. Answer only using provided context."},
                        {"role": "user", "content": f"Context: {' '.join(contexts[:2])}\n\nQuestion: {query}"}
                    ],
                    max_tokens=260,
                    temperature=0.0
                )
                
                answer = completion.choices[0].message.content
                gen_time = time.perf_counter() - start
                
                # Evaluate
                faith_result = faithfulness_score(answer, contexts)
                faith = faith_result['score']
                relev = relevance_score(query, answer)
                
                metrics.append({
                    "query": query,
                    "faithfulness": faith,
                    "relevancy": relev,
                    "latency_sec": gen_time
                })
                
                logger.info(f"  ✓ {query[:40]}... | Faith: {faith:.3f} | Relev: {relev:.3f}")
                
            except Exception as e:
                failed += 1
                logger.warning(f"  ✗ Failed: {str(e)[:60]}")
        
        if metrics:
            avg_faith = np.mean([m['faithfulness'] for m in metrics])
            avg_relev = np.mean([m['relevancy'] for m in metrics])
            avg_latency = np.mean([m['latency_sec'] for m in metrics])
            
            self.results["llm_comparison"][model_name] = {
                "avg_faithfulness": float(avg_faith),
                "avg_relevancy": float(avg_relev),
                "avg_latency_sec": float(avg_latency),
                "failed": failed,
                "success_rate": (len(metrics) / len(test_queries_subset))
            }
            
            logger.info(f"  Summary: Faith={avg_faith:.3f}, Relev={avg_relev:.3f}, Latency={avg_latency:.1f}s")
        
        return self.results["llm_comparison"].get(model_name, {})
    
    def test_retrieval_weights(self):
        """Test different semantic/BM25 weight combinations."""
        logger.info("\n" + "="*80)
        logger.info("🔍 RETRIEVAL WEIGHT OPTIMIZATION")
        logger.info("="*80)
        
        from generate_answer import generate_answer
        
        weight_combos = [
            (0.3, 0.7, "BM25-heavy"),
            (0.4, 0.6, "BM25-biased"),
            (0.5, 0.5, "Balanced"),
            (0.6, 0.4, "Semantic-biased"),
            (0.7, 0.3, "Semantic-heavy"),
        ]
        
        test_queries_subset = TEST_QUERIES[:4]  # Use 4 queries
        
        for sem_w, bm_w, label in weight_combos:
            logger.info(f"\n  {label}: {sem_w}/{bm_w}")
            metrics = []
            
            for query in test_queries_subset:
                try:
                    result = generate_answer(query, top_k=8, strategy='semantic',
                                            semantic_weight=sem_w, bm25_weight=bm_w)
                    answer = result['answer']
                    contexts = result['contexts']
                    
                    faith_result = faithfulness_score(answer, contexts)
                    faith = faith_result['score']
                    relev = relevance_score(query, answer)
                    
                    metrics.append({"faith": faith, "relev": relev})
                except Exception as e:
                    logger.warning(f"    Error: {str(e)[:40]}")
            
            if metrics:
                avg_faith = np.mean([m['faith'] for m in metrics])
                avg_relev = np.mean([m['relev'] for m in metrics])
                combined = (avg_faith + avg_relev) / 2
                
                self.results["weight_optimization"][label] = {
                    "weights": (sem_w, bm_w),
                    "avg_faithfulness": float(avg_faith),
                    "avg_relevancy": float(avg_relev),
                    "combined_score": float(combined)
                }
                
                logger.info(f"    Faith: {avg_faith:.3f} | Relev: {avg_relev:.3f} | Combined: {combined:.3f}")
    
    def test_chunking_strategies(self):
        """Compare different chunking strategies."""
        logger.info("\n" + "="*80)
        logger.info("📚 CHUNKING STRATEGY COMPARISON")
        logger.info("="*80)
        
        from generate_answer import generate_answer
        
        strategies = ['fixed', 'recursive', 'semantic']
        test_queries_subset = TEST_QUERIES[:3]  # Use 3 queries
        
        for strategy in strategies:
            logger.info(f"\n  {strategy.upper()}")
            metrics = []
            
            for query in test_queries_subset:
                try:
                    result = generate_answer(query, top_k=8, strategy=strategy,
                                            semantic_weight=0.5, bm25_weight=0.5)
                    answer = result['answer']
                    contexts = result['contexts']
                    
                    faith_result = faithfulness_score(answer, contexts)
                    faith = faith_result['score']
                    relev = relevance_score(query, answer)
                    latency = result['latency']['total_sec']
                    
                    metrics.append({"faith": faith, "relev": relev, "latency": latency})
                except Exception as e:
                    logger.warning(f"    Error: {str(e)[:40]}")
            
            if metrics:
                avg_faith = np.mean([m['faith'] for m in metrics])
                avg_relev = np.mean([m['relev'] for m in metrics])
                avg_latency = np.mean([m['latency'] for m in metrics])
                combined = (avg_faith + avg_relev) / 2
                
                self.results["chunk_strategy_comparison"][strategy] = {
                    "avg_faithfulness": float(avg_faith),
                    "avg_relevancy": float(avg_relev),
                    "avg_latency_sec": float(avg_latency),
                    "combined_score": float(combined)
                }
                
                logger.info(f"    Faith: {avg_faith:.3f} | Relev: {avg_relev:.3f} | Latency: {avg_latency:.1f}s")
    
    def run_all_optimizations(self):
        """Run complete optimization suite."""
        logger.info("\n" + "="*80)
        logger.info("🚀 COMPLETE SYSTEM OPTIMIZATION")
        logger.info("="*80)
        
        # 1. Test different LLM models
        logger.info("\n" + "="*80)
        logger.info("🤖 PHASE 1: LLM MODEL COMPARISON")
        logger.info("="*80)
        
        llm_models = [
            "mistralai/Mistral-7B-Instruct-v0.2",  # Current
            "meta-llama/Llama-2-7b-chat-hf",       # Alternative
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Fast
        ]
        
        for model in llm_models:
            try:
                self.test_llm_model(model, TEST_QUERIES[:2])  # Use 2 queries for speed
            except Exception as e:
                logger.error(f"LLM test failed: {str(e)[:80]}")
        
        # 2. Test retrieval weights
        self.test_retrieval_weights()
        
        # 3. Test chunking strategies
        self.test_chunking_strategies()
        
        # Identify best configuration
        self.identify_best_config()
    
    def identify_best_config(self):
        """Identify the best configuration across all tests."""
        logger.info("\n" + "="*80)
        logger.info("🏆 BEST CONFIGURATIONS BY CATEGORY")
        logger.info("="*80)
        
        # Best weights
        if self.results["weight_optimization"]:
            best_weights = max(self.results["weight_optimization"].items(),
                              key=lambda x: x[1]['combined_score'])
            logger.info(f"\nBest Weights: {best_weights[0]}")
            logger.info(f"  Score: {best_weights[1]['combined_score']:.3f}")
        
        # Best chunking
        if self.results["chunk_strategy_comparison"]:
            best_chunk = max(self.results["chunk_strategy_comparison"].items(),
                            key=lambda x: x[1]['combined_score'])
            logger.info(f"\nBest Chunking: {best_chunk[0]}")
            logger.info(f"  Score: {best_chunk[1]['combined_score']:.3f}")
        
        # Best LLM
        if self.results["llm_comparison"]:
            best_llm = max(self.results["llm_comparison"].items(),
                          key=lambda x: (x[1].get('avg_faithfulness', 0) + x[1].get('avg_relevancy', 0)) / 2)
            logger.info(f"\nBest LLM: {best_llm[0]}")
            logger.info(f"  Faithfulness: {best_llm[1].get('avg_faithfulness', 0):.3f}")
    
    def save_results(self):
        """Save optimization results."""
        output = Path("actual_optimization_results.json")
        with open(output, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"\n✅ Results saved to {output}")
        return output

if __name__ == "__main__":
    optimizer = RealOptimizer()
    optimizer.run_all_optimizations()
    optimizer.save_results()
    
    logger.info("\n" + "="*80)
    logger.info("✨ OPTIMIZATION COMPLETE")
    logger.info("Next: Review results and decide what actually improved")
    logger.info("="*80)
