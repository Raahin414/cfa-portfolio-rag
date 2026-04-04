import argparse
import os
import re
from functools import lru_cache
from statistics import mean

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

from generate_answer import generate_answer


MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_EVAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_LOCAL_FALLBACK_MODEL = "google/flan-t5-small"
model = SentenceTransformer(MODEL_NAME)

_REMOTE_QUESTION_GEN_DISABLED = False


load_dotenv()


def cosine_similarity(a, b):
    dot = float((a * b).sum())
    norm = float((a**2).sum()) ** 0.5 * float((b**2).sum()) ** 0.5
    if norm == 0.0:
        return 0.0
    return dot / norm


def _get_client_and_model(model_name=None):
    token = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    provider = os.getenv("HF_PROVIDER")
    model_name = (
        model_name
        or os.getenv("HF_EVAL_MODEL")
        or os.getenv("HF_GENERATION_MODEL")
        or DEFAULT_EVAL_MODEL
    )
    client = (
        InferenceClient(model=model_name, token=token, provider=provider, timeout=20)
        if provider
        else InferenceClient(model=model_name, token=token, timeout=20)
    )
    return client, model_name


def _extract_questions(raw_text):
    text = (raw_text or "").strip()
    if not text:
        return []

    # First, parse explicit numbered format even when the model returns one line.
    numbered = []
    for m in re.finditer(r"(?:^|\s)([1-3])\s*[\.)]\s*(.+?)(?=(?:\s[1-3]\s*[\.)]\s)|$)", text, flags=re.DOTALL):
        candidate = m.group(2).strip()
        candidate = re.sub(r"\s+", " ", candidate)
        if candidate:
            numbered.append(candidate)

    if len(numbered) >= 3:
        ordered = []
        for q in numbered:
            if "?" not in q:
                q = q.rstrip(". ") + "?"
            if q not in ordered:
                ordered.append(q)
        return ordered

    # Fallback: parse per line bullets/numbering.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cleaned = []
    for ln in lines:
        if ln.lower().startswith("return format"):
            continue
        ln = re.sub(r"^[-*\d\).\s]+", "", ln).strip()
        if not ln:
            continue
        if "?" not in ln:
            ln = ln.rstrip(". ") + "?"
        cleaned.append(re.sub(r"\s+", " ", ln))

    unique = []
    for q in cleaned:
        if q not in unique:
            unique.append(q)
    return unique


@lru_cache(maxsize=1)
def _get_local_seq2seq_model_and_tokenizer():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = os.getenv("LOCAL_FALLBACK_MODEL", DEFAULT_LOCAL_FALLBACK_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer


def _generate_with_local_model(prompt, max_new_tokens=96):
    model, tokenizer = _get_local_seq2seq_model_and_tokenizer()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def _generate_questions_with_local_llm(answer):
    prompt = (
        "You must generate exactly 3 questions.\n\n"
        "Answer:\n"
        f"{answer}\n\n"
        "Return format:\n\n"
        "1. Question one\n"
        "2. Question two\n"
        "3. Question three"
    )
    raw = _generate_with_local_model(prompt, max_new_tokens=96)
    return {
        "questions": _extract_questions(raw),
        "raw_response": raw,
        "backend": "local_llm_fallback",
    }


def generate_alternate_questions(answer, num_questions=3, model_name=None):
    global _REMOTE_QUESTION_GEN_DISABLED

    if _REMOTE_QUESTION_GEN_DISABLED:
        local = _generate_questions_with_local_llm(answer)
        questions = local["questions"]
        if len(questions) < num_questions:
            fallback = [
                "What is the main concept explained in the answer?",
                "Which key factors are described in the answer?",
                "How is the described concept applied in practice?",
            ]
            for q in fallback:
                if q not in questions:
                    questions.append(q)
                if len(questions) >= num_questions:
                    break
        return {
            "questions": questions[:num_questions],
            "raw_response": local.get("raw_response", ""),
            "backend": "local_llm_fallback",
        }

    client, chosen_model = _get_client_and_model(model_name=model_name)
    provider_l = (os.getenv("HF_PROVIDER") or "").lower()
    conversational_only_provider = provider_l in {"featherless-ai"}

    prompt = (
        "You must generate exactly 3 questions.\n\n"
        "Answer:\n"
        f"{answer}\n\n"
        "Return format:\n\n"
        "1. Question one\n"
        "2. Question two\n"
        "3. Question three"
    )

    raw = ""
    backend = "unknown"
    chat_error = None
    textgen_error = None
    questions = []
    for _ in range(3):
        try:
            completion = client.chat.completions.create(
                model=chosen_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Generate exactly 3 questions. "
                            "Output must contain 3 numbered lines only: 1., 2., 3."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=160,
                temperature=0.0,
            )
            raw = completion.choices[0].message.content.strip()
            backend = "chat.completions"
            questions = _extract_questions(raw)
            if len(questions) >= num_questions:
                break
        except Exception as e:
            chat_error = e
            err_txt = str(e).lower()
            if "402" in err_txt or "payment required" in err_txt or "not supported for task" in err_txt:
                _REMOTE_QUESTION_GEN_DISABLED = True
            if _REMOTE_QUESTION_GEN_DISABLED or conversational_only_provider:
                continue

        try:
            generated = client.text_generation(
                prompt=f"<s>[INST] {prompt} [/INST]",
                max_new_tokens=160,
                temperature=0.0,
                do_sample=False,
            )
            raw = generated.strip() if isinstance(generated, str) else str(generated).strip()
            backend = "text_generation"
            questions = _extract_questions(raw)
            if len(questions) >= num_questions:
                break
        except Exception as e:
            textgen_error = e
            err_txt = str(e).lower()
            if "402" in err_txt or "payment required" in err_txt or "not supported for task" in err_txt:
                _REMOTE_QUESTION_GEN_DISABLED = True

    if len(questions) < num_questions:
        try:
            local = _generate_questions_with_local_llm(answer)
            questions = local["questions"]
            raw = local["raw_response"]
            backend = "local_llm_fallback"
        except Exception as local_error:
            backend = (
                "fallback_template"
                f"(chat={type(chat_error).__name__ if chat_error else 'NA'},"
                f"text={type(textgen_error).__name__ if textgen_error else 'NA'},"
                f"local={type(local_error).__name__})"
            )

    if len(questions) < num_questions:
        fallback = [
            "What is the main concept explained in the answer?",
            "Which key factors are described in the answer?",
            "How is the described concept applied in practice?",
        ]
        for q in fallback:
            if q not in questions:
                questions.append(q)
            if len(questions) >= num_questions:
                break

    return {
        "questions": questions[:num_questions],
        "raw_response": raw,
        "backend": backend,
    }


def relevance_score(query, answer, return_details=False, model_name=None):
    if "information not found in dataset" in (answer or "").lower():
        details = {
            "generated_questions": [
                "What information is missing from the dataset for this query?",
                "Which dataset sources would be needed to answer this query?",
                "How can this query be reformulated to match portfolio-domain content?",
            ],
            "similarity_scores": [0.0, 0.0, 0.0],
            "average_score": 0.0,
            "method": "alternate_query_generation",
            "embedding_model": MODEL_NAME,
            "question_generation_backend": "refusal_guardrail",
        }
        if return_details:
            return details
        return 0.0

    generated = generate_alternate_questions(answer, num_questions=3, model_name=model_name)
    questions = generated["questions"]

    q_vec = model.encode(query, convert_to_numpy=True)
    similarity_scores = []
    for question in questions:
        g_vec = model.encode(question, convert_to_numpy=True)
        similarity_scores.append(float(cosine_similarity(q_vec, g_vec)))

    avg_score = float(mean(similarity_scores)) if similarity_scores else 0.0
    details = {
        "generated_questions": questions,
        "similarity_scores": similarity_scores,
        "average_score": avg_score,
        "method": "alternate_query_generation",
        "embedding_model": MODEL_NAME,
        "question_generation_backend": generated["backend"],
    }

    if return_details:
        return details
    return avg_score


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate query-answer relevance.")
    parser.add_argument("--query", required=True, help="Question to evaluate")
    parser.add_argument("--model", default=None, help="HF model for alternate question generation")
    return parser.parse_args()


def main():
    args = parse_args()
    result = generate_answer(args.query)
    report = relevance_score(
        result["query"],
        result["answer"],
        return_details=True,
        model_name=args.model,
    )

    print(f"Relevance score: {report['average_score']:.3f}")
    print("Generated questions:")
    for i, q in enumerate(report["generated_questions"], start=1):
        print(f"{i}. {q}")
    print("Similarity scores:")
    for i, s in enumerate(report["similarity_scores"], start=1):
        print(f"{i}. {s:.3f}")
    print("Answer preview:")
    print(result["answer"][:500])


if __name__ == "__main__":
    main()
