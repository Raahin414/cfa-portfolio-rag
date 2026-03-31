import argparse
import re
from difflib import SequenceMatcher

from generate_answer import generate_answer


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


def sentence_supported(sentence, contexts, threshold=0.45):
    best_ratio = 0.0
    context_sentences = []
    for ctx in contexts:
        context_sentences.extend(split_sentences(ctx))

    if not context_sentences:
        context_sentences = contexts

    for ctx_sentence in context_sentences:
        ratio = SequenceMatcher(None, sentence.lower(), (ctx_sentence or "").lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
    return best_ratio >= threshold, best_ratio


def faithfulness_score(answer, contexts, threshold=0.45):
    claims = split_sentences(answer)
    if not claims:
        return {
            "score": 0.0,
            "supported_claims": 0,
            "total_claims": 0,
        }

    supported = 0
    details = []
    for claim in claims:
        ok, ratio = sentence_supported(claim, contexts, threshold=threshold)
        supported += 1 if ok else 0
        details.append({"claim": claim, "supported": ok, "match_ratio": round(ratio, 3)})

    score = supported / len(claims)
    return {
        "score": score,
        "supported_claims": supported,
        "total_claims": len(claims),
        "details": details,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate answer faithfulness against retrieved context.")
    parser.add_argument("--query", required=True, help="Question to evaluate")
    parser.add_argument("--threshold", type=float, default=0.45, help="Claim-context similarity threshold")
    return parser.parse_args()


def main():
    args = parse_args()
    result = generate_answer(args.query)
    report = faithfulness_score(result["answer"], result["contexts"], threshold=args.threshold)

    print(f"Faithfulness score: {report['score']:.3f} ({report['supported_claims']}/{report['total_claims']})")
    for item in report.get("details", [])[:5]:
        print(f"- supported={item['supported']} ratio={item['match_ratio']}: {item['claim']}")


if __name__ == "__main__":
    main()
