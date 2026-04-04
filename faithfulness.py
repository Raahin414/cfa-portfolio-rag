import argparse
import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from generate_answer import generate_answer


DEFAULT_EVAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_LOCAL_FALLBACK_MODEL = "google/flan-t5-small"


load_dotenv()


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


def _get_client_and_model(model_name=None):
    token = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    provider = os.getenv("HF_PROVIDER")
    model_name = (
        model_name
        or os.getenv("HF_EVAL_MODEL")
        or os.getenv("HF_GENERATION_MODEL")
        or DEFAULT_EVAL_MODEL
    )
    client = InferenceClient(model=model_name, token=token, provider=provider) if provider else InferenceClient(model=model_name, token=token)
    return client, model_name


def _parse_support_label(raw_text):
    txt = (raw_text or "").strip().upper()
    if "NOT_SUPPORTED" in txt:
        return "NOT_SUPPORTED"
    if "SUPPORTED" in txt:
        return "SUPPORTED"
    return "NOT_SUPPORTED"


@lru_cache(maxsize=1)
def _get_local_seq2seq_model_and_tokenizer():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = os.getenv("LOCAL_FALLBACK_MODEL", DEFAULT_LOCAL_FALLBACK_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer


def _generate_with_local_model(prompt, max_new_tokens=8):
    model, tokenizer = _get_local_seq2seq_model_and_tokenizer()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def _verify_claim_with_local_llm(claim, retrieved_context):
    prompt = (
        "Determine if the claim is supported by the context. "
        "Reply with exactly one token: SUPPORTED or NOT_SUPPORTED.\n\n"
        f"Claim: {claim}\n\n"
        f"Context: {retrieved_context}\n\n"
        "Label:"
    )
    raw = _generate_with_local_model(prompt, max_new_tokens=4)
    label = _parse_support_label(raw)
    return {
        "claim": claim,
        "label": label,
        "supported": label == "SUPPORTED",
        "raw_response": raw,
        "backend": "local_llm_fallback",
    }


def _heuristic_support_label(claim, retrieved_context, min_overlap=0.25):
    # Fallback when remote evaluation APIs are unavailable.
    claim_tokens = set(re.findall(r"[a-zA-Z0-9]+", claim.lower()))
    context_tokens = set(re.findall(r"[a-zA-Z0-9]+", (retrieved_context or "").lower()))

    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with",
        "is", "are", "was", "were", "be", "by", "as", "at", "that", "this",
        "it", "from", "can", "will", "may", "if", "then", "than", "into", "about",
    }
    claim_tokens = {t for t in claim_tokens if len(t) > 2 and t not in stopwords}
    context_tokens = {t for t in context_tokens if len(t) > 2 and t not in stopwords}

    if not claim_tokens:
        return "NOT_SUPPORTED", 0.0

    overlap = len(claim_tokens & context_tokens) / len(claim_tokens)
    label = "SUPPORTED" if overlap >= min_overlap else "NOT_SUPPORTED"
    return label, float(overlap)


def verify_claim_with_llm(claim, retrieved_context, model_name=None):
    client, chosen_model = _get_client_and_model(model_name=model_name)

    prompt = (
        "You are an expert evaluator.\n\n"
        f"Claim:\n{claim}\n\n"
        f"Context:\n{retrieved_context}\n\n"
        "Is this claim supported by the context?\n\n"
        "Answer only:\n"
        "SUPPORTED\n"
        "NOT_SUPPORTED"
    )

    raw = ""
    backend = "unknown"
    try:
        completion = client.chat.completions.create(
            model=chosen_model,
            messages=[
                {
                    "role": "system",
                    "content": "Return only one label: SUPPORTED or NOT_SUPPORTED.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=8,
            temperature=0.0,
        )
        raw = completion.choices[0].message.content.strip()
        backend = "chat.completions"
    except Exception as chat_error:
        try:
            generated = client.text_generation(
                prompt=f"<s>[INST] {prompt} [/INST]",
                max_new_tokens=8,
                temperature=0.0,
                do_sample=False,
            )
            raw = generated.strip() if isinstance(generated, str) else str(generated).strip()
            backend = "text_generation"
        except Exception as textgen_error:
            try:
                verdict = _verify_claim_with_local_llm(claim, retrieved_context)
                verdict["fallback_reason"] = f"chat={type(chat_error).__name__}; text={type(textgen_error).__name__}"
                return verdict
            except Exception as local_error:
                label, overlap = _heuristic_support_label(claim, retrieved_context)
                return {
                    "claim": claim,
                    "label": label,
                    "supported": label == "SUPPORTED",
                    "raw_response": f"heuristic_overlap={overlap:.3f}",
                    "backend": "heuristic_fallback",
                    "fallback_reason": (
                        f"chat={type(chat_error).__name__}; text={type(textgen_error).__name__}; "
                        f"local={type(local_error).__name__}"
                    ),
                }

    label = _parse_support_label(raw)
    return {
        "claim": claim,
        "label": label,
        "supported": label == "SUPPORTED",
        "raw_response": raw,
        "backend": backend,
    }

def faithfulness_score(answer, contexts, threshold=0.45, model_name=None):
    claims = split_sentences(answer)
    if not claims:
        return {
            "score": 0.0,
            "supported_claims": 0,
            "total_claims": 0,
            "details": [],
            "method": "llm_claim_verification",
            "threshold": threshold,
        }

    retrieved_context = "\n\n".join(contexts[:5])
    supported = 0
    details = []
    for claim in claims:
        verdict = verify_claim_with_llm(claim, retrieved_context, model_name=model_name)
        supported += 1 if verdict["supported"] else 0
        details.append(verdict)

    score = supported / len(claims)
    return {
        "score": score,
        "supported_claims": supported,
        "total_claims": len(claims),
        "details": details,
        "method": "llm_claim_verification",
        "threshold": threshold,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate answer faithfulness against retrieved context.")
    parser.add_argument("--query", required=True, help="Question to evaluate")
    parser.add_argument("--threshold", type=float, default=0.45, help="Claim-context similarity threshold")
    parser.add_argument("--model", default=None, help="HF model for claim verification")
    return parser.parse_args()


def main():
    args = parse_args()
    result = generate_answer(args.query)
    report = faithfulness_score(
        result["answer"],
        result["contexts"],
        threshold=args.threshold,
        model_name=args.model,
    )

    print(f"Faithfulness score: {report['score']:.3f} ({report['supported_claims']}/{report['total_claims']})")
    for item in report.get("details", [])[:5]:
        print(f"- label={item['label']} supported={item['supported']}: {item['claim']}")


if __name__ == "__main__":
    main()
