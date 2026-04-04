# GitHub-Verified Master Notes (NLP Assignment 3 Mini Project)

## Purpose
This document is a GitHub-tracked, evidence-based project summary.
Only artifacts present on GitHub `main` are referenced.

## Student Context
- Name: Raahin Tajuddin
- ERP: 29207
- Project: RAG system for CFA/portfolio domain question answering

## Assignment Intent Covered
- Hybrid retrieval pipeline (semantic + BM25)
- Reranking stage after retrieval
- LLM-based answer generation grounded in retrieved context
- Automated multi-query evaluation (20-query submission run)
- Ablation and optimization evidence with measurable metrics
- Reproducible scripts and machine-readable outputs

## Core Implementation Files
- `hybrid_retrieval.py`: hybrid search logic and embedding-driven retrieval
- `reranker.py`: cross-encoder reranking stage
- `generate_answer.py`: end-to-end retrieval, rerank, generation, fallback, confidence outputs
- `faithfulness.py`: grounding/faithfulness metric logic
- `relevance.py`: query-answer relevance scoring

## Evaluation and Benchmark Files
- `submission_20_query_eval.py`: assignment-style 20-query evaluation script
- `submission_20_query_results.json`: final 20-query summary and per-query rows
- `retrieval_rerank_benchmark.py`: retrieval+rereank-only ablation script
- `retrieval_rerank_benchmark_results.json`: retrieval+rereank benchmark outputs
- `RETRIEVAL_RERANK_REPORT.md`: human-readable retrieval+rereank report
- `extensive_pipeline_benchmark.py`: staged broader benchmark script
- `extensive_benchmark_results.json`: broad benchmark machine output
- `ASSIGNMENT_BENCHMARK_REPORT.md`: benchmark report artifact

## Optimization and Supporting Analysis Files
- `OPTIMIZATION_REPORT.md`
- `COMPLETION_SUMMARY.md`
- `ASSIGNMENT_REPORT_FOCUS.md`
- `actual_optimization_results.json`
- `fixed_chunking_validation.json`
- `optimization_analysis_report.json`
- `hf_model_sweep_apr02_2026.md`
- `embedding_model_comparison_apr04_2026.json`

## Confirmed Final 20-Query Submission Metrics
Source: `submission_20_query_results.json`
- Queries: 20
- Strategy: semantic
- Weights: semantic=0.5, bm25=0.5
- top_k: 6
- Generation model: meta-llama/Meta-Llama-3-8B-Instruct
- Temperature: 0.0
- Max tokens: 300
- Avg faithfulness: 0.6031565656565656
- Avg relevance: 0.8751872252175656
- Avg latency sec: 3.418584204999206
- Success rate: 0.8

## Retrieval + Reranking Ablation Finding
Sources: `retrieval_rerank_benchmark_results.json`, `RETRIEVAL_RERANK_REPORT.md`
Best retrieval-side configuration:
- Chunking: semantic
- Weights: 0.6 / 0.4 (semantic / BM25)
- top_k: 6
- Avg post-rerank relevance: ~0.855
- Avg rerank lift: ~-0.003
- Avg total latency: ~0.360 sec

Interpretation:
- Retrieval setup was the primary quality driver in this corpus.
- Reranker average lift was near-neutral, but reranking remains part of the expected robust architecture.

## Embedding Model Comparison Decision
Source: `embedding_model_comparison_apr04_2026.json`
Compared models:
- BAAI/bge-small-en-v1.5
- sentence-transformers/all-MiniLM-L6-v2

Result:
- BGE objective (~0.6336) > all-MiniLM objective (~0.6084)
- Decision: keep BAAI/bge-small-en-v1.5 for current indexed corpus

## Practical Constraints Captured During Work
- External generation API availability can affect success rate and backend used.
- Fallback behavior exists to keep system operational when generation endpoint is unavailable.

## Recommended Report Claims (GitHub-Evidenced)
Use these claims safely because evidence exists in tracked artifacts:
- Hybrid retrieval + reranking architecture was implemented and tested.
- 20-query automated evaluation was completed with stored per-query outputs.
- Retrieval+rereank ablation was completed over multiple configurations.
- Embedding A/B comparison was completed and recorded.
- Configuration-level recommendations were derived from measured outputs.

## Non-GitHub Caution
Do not cite local-only files in final submission unless they are committed and pushed to GitHub first.
