import argparse

from sentence_transformers import SentenceTransformer

from generate_answer import generate_answer


MODEL_NAME = "BAAI/bge-small-en-v1.5"
model = SentenceTransformer(MODEL_NAME)


def cosine_similarity(a, b):
    dot = float((a * b).sum())
    norm = float((a**2).sum()) ** 0.5 * float((b**2).sum()) ** 0.5
    if norm == 0.0:
        return 0.0
    return dot / norm


def relevance_score(query, answer):
    q_vec = model.encode(query, convert_to_numpy=True)
    a_vec = model.encode(answer, convert_to_numpy=True)
    return cosine_similarity(q_vec, a_vec)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate query-answer relevance.")
    parser.add_argument("--query", required=True, help="Question to evaluate")
    return parser.parse_args()


def main():
    args = parse_args()
    result = generate_answer(args.query)
    score = relevance_score(result["query"], result["answer"])

    print(f"Relevance score: {score:.3f}")
    print("Answer preview:")
    print(result["answer"][:500])


if __name__ == "__main__":
    main()
