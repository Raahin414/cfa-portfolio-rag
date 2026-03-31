import json
import argparse
import hashlib
import re
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_DATASET = "portfolio_dataset_final.json"
MIN_CHUNK_WORDS = 30
EMBEDDING_BATCH_SIZE = 64


def load_dataset(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Dataset must be a list of documents")

    return data


def normalize_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_id(strategy_name, source, chunk_index, text):
    digest = hashlib.sha1(f"{strategy_name}|{source}|{chunk_index}|{text}".encode("utf-8")).hexdigest()
    return digest


# =========================
# 🔹 CHUNKING METHODS
# =========================

# 1. FIXED CHUNKING
def fixed_chunk(text, chunk_size=400, overlap=50):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)

    return chunks


# 2. RECURSIVE CHUNKING (paragraph-based)
def recursive_chunk(text, max_words=400):
    if max_words <= 0:
        raise ValueError("max_words must be > 0")

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = []
    current_words = 0

    for para in paragraphs:
        words = para.split()

        # If a single paragraph is too long, split it directly.
        if len(words) > max_words:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_words = 0

            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i:i + max_words]))
            continue

        if current_words + len(words) <= max_words:
            current.append(para)
            current_words += len(words)
        else:
            chunks.append(" ".join(current))
            current = [para]
            current_words = len(words)

    if current:
        chunks.append(" ".join(current))

    return chunks


# 3. SEMANTIC CHUNKING (simple version)
def semantic_chunk(text, sentences_per_chunk=5):
    if sentences_per_chunk <= 0:
        raise ValueError("sentences_per_chunk must be > 0")

    # Safer sentence split than plain ". ", keeps punctuation boundaries.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = []

    for sentence in sentences:
        current.append(sentence)
        if len(current) >= sentences_per_chunk:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks


# =========================
# 🔹 PROCESS FUNCTION
# =========================

def process_data(data, model, strategy_name, chunk_func, output_dir="."):
    all_chunks = []
    pending_texts = []
    pending_meta = []

    output_path = Path(output_dir) / f"{strategy_name}_chunks.json"

    def flush_embeddings():
        if not pending_texts:
            return

        embeddings = model.encode(
            pending_texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        for emb, meta in zip(embeddings, pending_meta):
            all_chunks.append({
                "id": chunk_id(strategy_name, meta["source"], meta["chunk_index"], meta["text"]),
                "text": meta["text"],
                "embedding": emb.tolist(),
                "metadata": {
                    "title": meta["title"],
                    "topic": meta["topic"],
                    "source": meta["source"],
                    "strategy": strategy_name,
                    "chunk_index": meta["chunk_index"],
                    "word_count": meta["word_count"],
                }
            })

        pending_texts.clear()
        pending_meta.clear()

    for doc in tqdm(data, desc=f"Chunking {strategy_name}"):
        content = doc.get("content", "")
        if not content:
            continue

        chunks = chunk_func(content)

        for idx, chunk in enumerate(chunks):
            chunk = normalize_text(chunk)
            word_count = len(chunk.split())

            if word_count < MIN_CHUNK_WORDS:
                continue

            pending_texts.append(chunk)
            pending_meta.append({
                "text": chunk,
                "title": doc.get("title", ""),
                "topic": doc.get("topic", ""),
                "source": doc.get("source", doc.get("url", "")),
                "chunk_index": idx,
                "word_count": word_count,
            })

            if len(pending_texts) >= EMBEDDING_BATCH_SIZE:
                flush_embeddings()

    flush_embeddings()

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    if all_chunks:
        dims = len(all_chunks[0]["embedding"])
    else:
        dims = 0

    print(f"\nSaved {output_path.name}: {len(all_chunks)} chunks | embedding_dim={dims}")

    return {
        "strategy": strategy_name,
        "count": len(all_chunks),
        "embedding_dim": dims,
        "output": str(output_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Chunk and embed portfolio dataset for RAG.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Input dataset JSON path")
    parser.add_argument("--output-dir", default=".", help="Output directory for chunks JSON files")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["fixed", "recursive", "semantic"],
        choices=["fixed", "recursive", "semantic"],
        help="Chunking strategies to run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_dataset(args.dataset)

    model = SentenceTransformer(MODEL_NAME)

    strategies = {
        "fixed": fixed_chunk,
        "recursive": recursive_chunk,
        "semantic": semantic_chunk,
    }

    summaries = []
    for name in args.strategies:
        summaries.append(process_data(data, model, name, strategies[name], output_dir=args.output_dir))

    print("\nRun summary:")
    for item in summaries:
        print(f"- {item['strategy']}: {item['count']} chunks -> {item['output']}")


if __name__ == "__main__":
    main()
