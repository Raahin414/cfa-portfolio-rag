# Retrieval + Reranking Benchmark Report

- Queries: 20
- Configurations: 27

## Best Configuration
- Chunking: semantic
- Weights: semantic=0.6, bm25=0.4
- top_k: 6

## Best Metrics
- Avg pre-rerank relevance: 0.857
- Avg post-rerank relevance: 0.855
- Avg rerank lift: -0.003
- Avg total latency (s): 0.360
- Objective: 0.628

## Top 5
| Rank | Strategy | Weights | top_k | PostRel | Lift | Latency(s) | Objective |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | semantic | 0.6/0.4 | 6 | 0.855 | -0.003 | 0.360 | 0.628 |
| 2 | semantic | 0.6/0.4 | 8 | 0.855 | -0.001 | 0.415 | 0.626 |
| 3 | semantic | 0.5/0.5 | 8 | 0.855 | -0.001 | 0.420 | 0.626 |
| 4 | semantic | 0.5/0.5 | 6 | 0.855 | -0.003 | 0.410 | 0.626 |
| 5 | semantic | 0.5/0.5 | 10 | 0.854 | -0.003 | 0.477 | 0.622 |