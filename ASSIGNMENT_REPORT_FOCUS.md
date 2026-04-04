# CFA Portfolio RAG - Report-Focused Evaluation Summary

Date: 2026-04-02

## What was tested (extensive, assignment-aligned)
- Chunking strategies: fixed, recursive, semantic
- Hybrid retrieval weights: semantic/bm25 in {0.5/0.5, 0.6/0.4, 0.4/0.6}
- Retrieval depth: top_k in {6, 8, 10}
- Reranking: cross-encoder/ms-marco-MiniLM-L-6-v2 (always enabled)
- Generation models: meta-llama/Meta-Llama-3-8B-Instruct, Qwen/Qwen2.5-7B-Instruct
- Generation params: temperature in {0.0, 0.1}, max_tokens in {220, 260, 300}
- Evaluation metrics: faithfulness, relevance, latency, generation success rate

Artifacts generated:
- `extensive_benchmark_results.json`
- `ASSIGNMENT_BENCHMARK_REPORT.md`
- `retrieval_rerank_benchmark_results.json`
- `RETRIEVAL_RERANK_REPORT.md`

## Important runtime caveat
During long benchmark execution, Hugging Face credits were intermittently depleted (HTTP 402), so some later generation runs fell back to context-only output.

For report integrity:
- Use generation conclusions from runs with successful API responses.
- Use retrieval/reranking ablation as stable evidence (no LLM-credit dependency).

## Reliable findings

### 1) Best chunking strategy
Semantic chunking was consistently best in both full-pipeline successful windows and retrieval+rereank-only ablation.

Evidence:
- Retrieval+rereank best config: semantic, 0.6/0.4, top_k=6
- Full-pipeline successful phase-A top config: semantic, 0.5/0.5, top_k=6

### 2) Best retrieval settings (report recommendation)
Primary recommendation:
- strategy: semantic
- semantic_weight: 0.5
- bm25_weight: 0.5
- top_k: 6

Latency-focused alternative:
- strategy: semantic
- semantic_weight: 0.6
- bm25_weight: 0.4
- top_k: 6

### 3) Best generation model under current constraints
Model availability and quality tests showed:
- meta-llama/Meta-Llama-3-8B-Instruct: best quality
- Qwen/Qwen2.5-7B-Instruct: faster, slightly lower quality

Recommended for assignment quality:
- model: meta-llama/Meta-Llama-3-8B-Instruct
- temperature: 0.0
- max_tokens: 260 to 300

## Quantitative summary for report

### Full-pipeline (successful phase-A window, 8 representative queries)
Best successful config:
- model: meta-llama/Meta-Llama-3-8B-Instruct
- strategy: semantic
- weights: 0.5/0.5
- top_k: 6
- temperature: 0.0
- max_tokens: 260

Metrics:
- avg faithfulness: 0.900
- avg relevance: 0.855
- avg latency: 3.701s
- generation success rate: 1.000

### Retrieval+rereank ablation (20 queries, no LLM dependency)
Best config:
- strategy: semantic
- weights: 0.6/0.4
- top_k: 6

Metrics:
- avg pre-rerank relevance: 0.857
- avg post-rerank relevance: 0.855
- avg rerank lift: -0.003
- avg retrieval+rereank latency: 0.360s

Interpretation:
- Reranker impact on this corpus is small/neutral on average with current candidate set.
- Retrieval configuration dominates downstream quality differences.

## Assignment-ready conclusions
1. Chunking ablation conclusion:
   - semantic chunking performs best overall.
2. Hybrid search conclusion:
   - balanced (0.5/0.5) and semantic-biased (0.6/0.4) are both strong; choose 0.5/0.5 for quality balance, 0.6/0.4 for slightly faster retrieval profile.
3. Reranking conclusion:
   - keep reranker for robustness, but measured average lift is small; note this in ablation discussion.
4. Generation model conclusion:
   - Llama-3-8B gives best answer quality under assignment prompts.
5. Final recommended production combo:
   - model: meta-llama/Meta-Llama-3-8B-Instruct
   - chunking: semantic
   - semantic/bm25: 0.5/0.5
   - top_k: 6
   - temperature: 0.0
   - max_tokens: 300

## Suggested text for your report (short form)
"We performed an extensive staged ablation across chunking, retrieval weights, top-k, reranking, and generation model/parameters. Semantic chunking consistently outperformed fixed and recursive chunking. The best full-pipeline configuration during successful inference windows used Meta-Llama-3-8B-Instruct with semantic chunking, balanced hybrid retrieval (0.5/0.5), top-k=6, temperature 0.0, and max_tokens 260-300, achieving strong faithfulness and relevance. Retrieval+rereanking ablation over 20 queries confirmed semantic chunking as best and showed that retrieval settings contributed more to quality variance than reranker lift in our corpus."