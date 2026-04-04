# Assignment Benchmark Report (Extensive Pipeline Evaluation)

## Scope
- Retrieval: hybrid semantic + BM25 weights, top-k, chunking strategy
- Reranking: CrossEncoder included in normal pipeline
- Generation: model, temperature, max_tokens
- Metrics: faithfulness, relevance, latency, generation success rate

## Best Final Configuration
- Model: meta-llama/Meta-Llama-3-8B-Instruct
- Chunking strategy: semantic
- Weights: semantic=0.5, bm25=0.5
- top_k: 8
- temperature: 0.0
- max_tokens: 300

## Best Final Metrics (20-query set)
- Avg faithfulness: 0.924
- Avg relevance: 0.854
- Avg latency (s): 1.639
- Generation success rate: 0.250
- Objective score: 0.845

## Finalist Comparison
| Rank | Model | Strategy | Weights | top_k | Temp | Tokens | Faith | Rel | Latency(s) | Success | Objective |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | meta-llama/Meta-Llama-3-8B-Instruct | semantic | 0.5/0.5 | 8 | 0.0 | 300 | 0.924 | 0.854 | 1.639 | 0.250 | 0.845 |
| 2 | Qwen/Qwen2.5-7B-Instruct | semantic | 0.6/0.4 | 6 | 0.1 | 300 | 0.897 | 0.857 | 1.118 | 0.250 | 0.840 |
| 3 | Qwen/Qwen2.5-7B-Instruct | semantic | 0.5/0.5 | 8 | 0.1 | 220 | 0.869 | 0.853 | 1.505 | 0.400 | 0.816 |
| 4 | Qwen/Qwen2.5-7B-Instruct | semantic | 0.5/0.5 | 8 | 0.1 | 300 | 0.806 | 0.873 | 1.723 | 0.500 | 0.786 |

## Ablation Insights
- Chunking strategy impact is reflected by holding generation settings fixed in phase A.
- Retrieval weight impact is reflected by comparing 0.5/0.5 vs 0.6/0.4.
- Generation model/temperature/token impact is reflected in phase B and validated in phase C.

## Notes
- This report is generated from actual runs against the deployed Pinecone index.
- Full machine-readable results are in `extensive_benchmark_results.json`.