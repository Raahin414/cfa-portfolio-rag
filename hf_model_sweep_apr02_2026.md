# Hugging Face Model Sweep (Assignment Conditions)

Date: 2026-04-02

Setup:
- Retrieval: fixed chunking, top_k=8, semantic/bm25=0.5/0.5
- Rerank: cross-encoder/ms-marco-MiniLM-L-6-v2
- Prompting: grounded CFA prompt with context-only instruction and citations
- Evaluation: 6 CFA portfolio queries

## Availability Check
- meta-llama/Meta-Llama-3-8B-Instruct: available and runnable
- Qwen/Qwen2.5-7B-Instruct: available and runnable
- mistralai/Mistral-7B-Instruct-v0.2: not runnable with enabled providers (BadRequest)

## Results

| Model | Success | Avg Faithfulness | Avg Relevancy | Avg Latency |
|---|---:|---:|---:|---:|
| meta-llama/Meta-Llama-3-8B-Instruct | 6/6 | 0.824 | 0.880 | 2.99s |
| Qwen/Qwen2.5-7B-Instruct | 6/6 | 0.801 | 0.868 | 1.91s |
| mistralai/Mistral-7B-Instruct-v0.2 | 0/6 | N/A | N/A | N/A |

## Recommendation
- Best quality under current assignment constraints: meta-llama/Meta-Llama-3-8B-Instruct.
- Fastest available option with near-quality performance: Qwen/Qwen2.5-7B-Instruct.
- Keep provider unset in environment to allow automatic routing to an enabled provider.
