import argparse
import os
import time
import logging
from functools import lru_cache

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from hybrid_retrieval import hybrid_search
from reranker import rerank_with_scores

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


DEFAULT_QUERY = "How to diversify a portfolio to reduce risk?"
DEFAULT_HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_LOCAL_FALLBACK_MODEL = "google/flan-t5-small"
DEFAULT_MAX_TOKENS = 260
DEFAULT_TEMPERATURE = 0.0


load_dotenv()


def build_prompt(query, contexts):
    """Build improved prompt with better structure and guidance."""
    context_block = "\n\n".join([f"Context {i + 1}: {c}" for i, c in enumerate(contexts)])
    
    # Improved prompt with better instructions
    return (
        "You are an expert finance educator specializing in portfolio management (CFA curriculum).\n\n"
        "STRICT RULES:\n"
        "1. Answer ONLY from the provided context - do NOT add external knowledge\n"
        "2. If information is not in context, respond exactly: 'Information not found in dataset'\n"
        "3. For definitions: provide the term, formal definition, and practical examples when relevant\n"
        "4. For calculations: show the formula AND explain each component\n"
        "5. Organize using bullet points for lists or numbered points for steps\n"
        "6. If contexts provide conflicting info, state both viewpoints explicitly\n"
        "7. MANDATORY: End with a complete, grammatically correct sentence\n"
        "8. Add inline citations [Context 1], [Context 2] to support key claims\n"
        "9. Be concise but complete - under 150 words unless the topic requires more\n"
        "10. For time-based concepts: make chronological relationships explicit\n\n"
        f"Question: {query}\n\n"
        f"PROVIDED CONTEXT:\n{context_block}\n\n"
        "Answer (grounded ONLY in provided context, with citations):"
    )


def hf_generate(prompt, model_name=None, max_tokens=DEFAULT_MAX_TOKENS, temperature=DEFAULT_TEMPERATURE):
    hf_token = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    model_name = model_name or os.getenv("HF_GENERATION_MODEL", DEFAULT_HF_MODEL)
    provider = os.getenv("HF_PROVIDER")
    client = InferenceClient(model=model_name, token=hf_token, provider=provider) if provider else InferenceClient(model=model_name, token=hf_token)

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer ONLY from provided context. "
                        "If missing, reply exactly: 'Information not found in dataset'. "
                        "No external knowledge."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        result = completion.choices[0].message.content.strip()
        logger.debug(f"chat.completions succeeded, result length: {len(result)}")
        return result
    except Exception as e:
        logger.warning(f"chat.completions failed: {type(e).__name__}: {str(e)[:200]}")

    try:
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        generated = client.text_generation(
            prompt=formatted_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
        )
        if isinstance(generated, str):
            result = generated.strip()
        else:
            result = str(generated).strip()
        logger.debug(f"text_generation succeeded, result length: {len(result)}")
        return result
    except Exception as e:
        logger.error(f"text_generation also failed: {type(e).__name__}: {str(e)[:200]}")
        return None


@lru_cache(maxsize=1)
def _get_local_seq2seq_model_and_tokenizer():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    local_model_name = os.getenv("LOCAL_FALLBACK_MODEL", DEFAULT_LOCAL_FALLBACK_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(local_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(local_model_name)
    return model, tokenizer


def local_llm_generate(prompt, max_tokens=DEFAULT_MAX_TOKENS):
    try:
        model, tokenizer = _get_local_seq2seq_model_and_tokenizer()
        local_prompt = (
            "Answer the finance question using only the provided context. "
            "If information is missing, reply exactly: Information not found in dataset.\n\n"
            f"{prompt}"
        )
        inputs = tokenizer(local_prompt, return_tensors="pt", truncation=True, max_length=1024)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    except Exception as e:
        logger.error(f"local_llm_generate failed: {type(e).__name__}: {str(e)[:200]}")
        return None


def compute_answer_confidence(answer, contexts, query=""):
    """
    Compute confidence score for an answer (0-1).
    
    Factors:
    - Citations present (critical for grounded generation)
    - Answer length vs context length (specific vs generic)
    - Sentence completeness
    - Relevant keywords from query present in answer
    """
    
    # Factor 1: Citation presence (40% weight) - most important for grounded RAG
    citation_score = 0.4 if "[Context" in answer else 0.0
    
    # Factor 2: Sentence completeness (20% weight)
    completeness = 0.2 if answer.strip().endswith(('.', '?', '!')) else 0.0
    
    # Factor 3: Length appropriateness (20% weight)
    # Good answers are not too short, not too long
    answer_words = len(answer.split())
    context_words = sum(len(c.split()) for c in contexts)
    
    if 50 < answer_words < 300:  # Sweet spot for portfolio Q&A
        length_score = 0.2
    elif 20 < answer_words < 500:
        length_score = 0.1
    else:
        length_score = 0.0
    
    # Factor 4: Query relevance (20% weight)
    # Check if key query terms appear in answer
    query_keywords = set(query.lower().split()) if query else set()
    answer_keywords = set(answer.lower().split())
    
    overlap = len(query_keywords & answer_keywords)
    relevance_score = min(0.2 * (overlap / max(len(query_keywords), 1)), 0.2)
    
    # Combined confidence
    confidence = citation_score + completeness + length_score + relevance_score
    return min(confidence, 1.0)


def fallback_generate(query, contexts):
    """Generate better fallback answer when HF API fails."""
    if not contexts:
        return "I do not have enough retrieved context to answer this question."
    
    # Combine contexts more intelligently
    combined = " ".join([c if isinstance(c, str) else str(c) for c in contexts])
    
    # Clean common noise
    combined = combined.replace("Learning Module", "").replace("CONSTRAINTS describe", "").strip()
    
    # Extract best portion (first 200 words usually have most relevant context)
    words = combined.split()[:180]
    summary = " ".join(words)
    
    # Add trailing context indicators
    if len(words) == 180:
        summary = summary.rstrip() + " [from retrieved contexts]"
    
    logger.info(f"Using fallback answer (HF API unavailable, confidence: lower)")
    return summary


def generate_answer(
    query,
    top_k=8,
    strategy="fixed",
    semantic_weight=0.5,
    bm25_weight=0.5,
    generation_model=None,
    temperature=DEFAULT_TEMPERATURE,
    max_tokens=DEFAULT_MAX_TOKENS,
):
    start = time.perf_counter()
    retrieval = hybrid_search(
        query,
        top_k=top_k or 8,
        strategy=strategy,
        semantic_weight=semantic_weight,
        bm25_weight=bm25_weight,
        return_debug=True,
    )
    docs = retrieval["docs"]
    hits = retrieval.get("hits", [])
    hit_by_text = {h.get("text", ""): h for h in hits if h.get("text")}

    rerank_start = time.perf_counter()
    ranked_docs = rerank_with_scores(query, docs)
    rerank_latency = time.perf_counter() - rerank_start
    selected_ranked = ranked_docs[:5]
    selected = [item["doc"] for item in selected_ranked]

    sources = []
    rerank_scores = [item["score"] for item in selected_ranked]
    for item in selected_ranked:
        text = item["doc"]
        hit = hit_by_text.get(text, {})
        sources.append(
            {
                "source": hit.get("source", ""),
                "topic": hit.get("topic", ""),
                "retrieval_score": float(hit.get("score", 0.0)),
                "rerank_score": float(item["score"]),
                "preview": text[:260].replace("\n", " "),
            }
        )

    prompt = build_prompt(query, selected)

    generation_start = time.perf_counter()
    llm_answer = None
    try:
        llm_answer = hf_generate(
            prompt,
            model_name=generation_model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as e:
        logger.error(f"hf_generate raised exception: {type(e).__name__}: {str(e)[:200]}")

    local_answer = None
    if not llm_answer:
        local_answer = local_llm_generate(prompt, max_tokens=max_tokens)

    generation_latency = time.perf_counter() - generation_start
    answer = llm_answer if llm_answer else (local_answer if local_answer else fallback_generate(query, selected))
    backend = (
        "huggingface_inference_api"
        if llm_answer
        else ("local_llm_fallback" if local_answer else "fallback_context_only")
    )
    
    # Compute answer confidence score
    answer_confidence = compute_answer_confidence(answer, selected, query)

    return {
        "query": query,
        "answer": answer,
        "contexts": selected,
        "latency": {
            "retrieval_sec": retrieval["latency_sec"],
            "rerank_sec": rerank_latency,
            "generation_sec": generation_latency,
            "total_sec": time.perf_counter() - start,
        },
        "strategy": strategy,
        "weights": {
            "semantic": semantic_weight,
            "bm25": bm25_weight,
        },
        "generation_backend": backend,
        "generation_config": {
            "model": generation_model or os.getenv("HF_GENERATION_MODEL", DEFAULT_HF_MODEL),
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        "sources": sources,
        "confidence": {
            "answer_confidence_score": float(answer_confidence),
            "rerank_mean": (sum(rerank_scores) / len(rerank_scores)) if rerank_scores else 0.0,
            "num_supporting_contexts": len(selected),
            "has_citations": "[Context" in answer,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate an answer from retrieved and reranked chunks.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="User question")
    parser.add_argument("--top-k", type=int, default=8, help="Number of docs to retrieve before rerank")
    parser.add_argument("--strategy", default="fixed", choices=["fixed", "recursive", "semantic"], help="Chunking strategy namespace")
    parser.add_argument("--semantic-weight", type=float, default=0.5, help="Weight for semantic retrieval")
    parser.add_argument("--bm25-weight", type=float, default=0.5, help="Weight for BM25 retrieval")
    parser.add_argument("--model", default=None, help="HF generation model (overrides env HF_GENERATION_MODEL)")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Generation temperature")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Maximum generation tokens")
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    result = generate_answer(
        args.query,
        top_k=args.top_k,
        strategy=args.strategy,
        semantic_weight=args.semantic_weight,
        bm25_weight=args.bm25_weight,
        generation_model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print("Question:")
    print(result["query"])
    print("\nAnswer:")
    print(result["answer"])
    print("\nTop Contexts Used:")
    for i, ctx in enumerate(result["contexts"], start=1):
        preview = ctx[:220].replace("\n", " ")
        print(f"{i}. {preview}...")
    print("\nLatency:")
    print(f"- retrieval_sec: {result['latency']['retrieval_sec']:.3f}")
    print(f"- rerank_sec: {result['latency']['rerank_sec']:.3f}")
    print(f"- generation_sec: {result['latency']['generation_sec']:.3f}")
    print(f"- total_sec: {result['latency']['total_sec']:.3f}")
    print(f"- generation_backend: {result['generation_backend']}")


if __name__ == "__main__":
    main()
