# CFA Portfolio Management RAG

Grounded Q&A system over CFA Kaplan portfolio management materials using retrieval-augmented generation (RAG).

## Features

- **Hybrid Retrieval**: Combines semantic embeddings (BAAI/bge-small-en-v1.5) with BM25 lexical search
- **Reranking**: Cross-encoder-based reranking for improved relevance
- **Grounded Generation**: Mistral 7B LLM with strict extractive prompting to avoid hallucinations
- **Evaluation Metrics**: Faithfulness (claim-to-context matching) and relevance (query-answer similarity)
- **Interactive UI**: Streamlit interface with confidence scores and source citations

## Architecture

```
Query
  ↓
[Hybrid Retrieval] → Semantic (BGE) + BM25 (0.5/0.5 weights)
  ↓
[Top-8 Documents] → [Reranking] → Top-5 Most Relevant
  ↓
[Grounded Generation] → Mistral 7B (temperature=0°, extractive style)
  ↓
Answer + Sources + Confidence Metrics
```

### Models & Services

- **Embeddings**: BAAI/bge-small-en-v1.5 (384-dimensional)
- **LLM**: mistralai/Mistral-7B-Instruct-v0.2 via HuggingFace Inference API
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **Vector DB**: Pinecone (serverless, cosine metric)

### Performance (Validated on 15 CFA Benchmark Queries)

- **Avg Faithfulness**: 0.901 (sentence-level claim grounding, threshold ≥0.45)
- **Avg Relevance**: 0.886 (query-answer semantic similarity)
- **Avg Latency**: 5.6 seconds (end-to-end retrieval + generation)
- **Dataset**: 1,097 documents from CFA Kaplan materials + portfolio examples

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and add your credentials:

```bash
cp .env.example .env
```

Edit `.env` with:
- **PINECONE_API_KEY**: Your Pinecone API key
- **PINECONE_INDEX**: Index name (default: `portfolio-rag-384`)
- **HF_API_KEY**: Your HuggingFace Token

```env
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_INDEX=portfolio-rag-384
HF_API_KEY=your_hf_api_key_here
HF_GENERATION_MODEL=mistralai/Mistral-7B-Instruct-v0.2
HF_PROVIDER=featherless-ai
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Usage

### Interactive Q&A (Streamlit UI)

1. Enter a portfolio management question
2. Adjust retrieval settings (strategy, weights, top-k)
3. View grounded answer, confidence score, and source citations

Example questions:
- "What is the efficient frontier?"
- "How does diversification reduce risk?"
- "What is the Sharpe ratio?"

### Command-Line Queries

```bash
# Generate answer to a question
python generate_answer.py --query "What is asset allocation?" --strategy semantic --semantic-weight 0.5 --bm25-weight 0.5

# Evaluate faithfulness
python faithfulness.py --query "What is portfolio rebalancing?"

# Check relevance
python relevance.py --query "How does CAPM relate to expected return?"
```

### Evaluation & Benchmarking

```bash
# Ablation study across chunking strategies (fixed, recursive, semantic)
python ablation_study.py --output ablation_results.json

# Optimize hybrid retrieval weights
python retrieval_optimization.py --strategy semantic --output retrieval_optimization_results.json

# Final validation on 15 benchmark queries
python final_validation.py \
    --strategy semantic \
    --semantic-weight 0.5 \
    --bm25-weight 0.5 \
    --output final_validation_report.json
```

## Data Preparation

### Adding Your Own PDFs

1. Place PDFs in the `data/` folder
2. Update chunking and embedding if needed:
   ```bash
   python chunking_embedding.py --dataset your_dataset.json --output-dir .
   ```
3. Upload embeddings to Pinecone:
   ```bash
   python upload_to_pinecone.py --chunk-files your_fixed_chunks.json your_recursive_chunks.json your_semantic_chunks.json
   ```

## Project Structure

```
cfa-portfolio-rag-deploy-ready/
├── app.py                           # Streamlit UI
├── generate_answer.py               # Core pipeline (reteval → rerank → generate)
├── hybrid_retrieval.py              # Dual-path semantic + BM25 search
├── reranker.py                      # Cross-encoder reranking with scores
├── chunking_embedding.py            # 3-strategy document chunking + BGE embeddings
├── upload_to_pinecone.py            # Upload embeddings to vector DB
├── faithfulness.py                  # Evaluate claim-to-context grounding
├── relevance.py                     # Query-answer semantic similarity
├── ablation_study.py                # Compare chunking strategies
├── retrieval_optimization.py        # Tune semantic/BM25 weights
├── final_validation.py              # Benchmark on 15 queries
├── requirements.txt                 # Python dependencies
├── .env.example                     # Template for credentials
├── .gitignore                       # Ignore sensitive/large files
├── data/                            # User-provided PDFs (not committed)
└── outputs/                         # Evaluation reports & results
```

## Deployment to Hugging Face Spaces

### 1. Prepare Repository

```bash
git init
git add -A
git commit -m "Initial commit: CFA Portfolio RAG"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/cfa-portfolio-rag.git
git push -u origin main
```

### 2. Create HF Spaces App

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **Create new Space**
3. Select **Streamlit** as the SDK
4. Connect your GitHub repository
5. Add secrets in Space settings:
   - `PINECONE_API_KEY`
   - `HF_API_KEY`
   - `PINECONE_REGION`
   - `PINECONE_CLOUD`
   - `PINECONE_INDEX`

### 3. App Deployment

Spaces auto-deploys on each GitHub push. Public URL appears in Space details.

## Development Workflow

### Local Testing Checklist

- [ ] `.env.example` has placeholders (no real keys)
- [ ] `.gitignore` excludes `.env`, `*_chunks.json`, `*.log`
- [ ] All scripts run without errors: `python -m py_compile *.py`
- [ ] Streamlit app launches: `streamlit run app.py`
- [ ] Sample query generates answer with confidence metrics
- [ ] No copyrighted PDFs or sensitive files in repo

### For Contributors

1. Clone the repo
2. Create `.env` from `.env.example` with your credentials
3. Update `.gitignore` if adding new temporary file types
4. Test locally before pushing
5. Ensure no real API keys are committed

## Troubleshooting

### Missing `.env` or Credentials

**Error**: `PINECONE_API_KEY is missing`  
**Solution**: Create `.env` file with API keys from `.env.example`

### Embedding Dimension Mismatch

**Error**: `Embedding dimension mismatch across files`  
**Solution**: Ensure all chunk files use BAAI/bge-small-en-v1.5 (384-dim)

### Pinecone Connection Issues

**Error**: `Failed to connect to Pinecone index`  
**Solution**: 
- Verify `PINECONE_API_KEY` and `PINECONE_INDEX` in `.env`
- Check index exists in Pinecone dashboard
- Ensure namespace exists (usually auto-created)

### HuggingFace API Timeouts

**Error**: `InferenceClient timeout or provider not found`  
**Solution**:
- Set `HF_PROVIDER=featherless-ai` in `.env` for proper routing
- Ensure HF token has inference access

## References

- [Pinecone Documentation](https://docs.pinecone.io/)
- [HuggingFace Inference API](https://huggingface.co/inference-api)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Sentence Transformers](https://www.sbert.net/)

## License

This project is provided as-is for educational purposes. Ensure compliance with CFA Institute guidelines when using Kaplan materials.

---

**Last Updated**: March 2025  
**Status**: Production-ready for Hugging Face Spaces deployment
