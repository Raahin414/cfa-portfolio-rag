import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from pinecone import Pinecone, ServerlessSpec


DEFAULT_INDEX_NAME = "portfolio-rag"
DEFAULT_CHUNK_FILES = [
    "fixed_chunks.json",
    "recursive_chunks.json",
    "semantic_chunks.json",
]
DEFAULT_BATCH_SIZE = 50

NOISE_PREFIXES = (
    "investopedia /",
    "get personalized, ai-powered answers",
)

NOISE_TOKENS = (
    "chartered market technician",
    "fact-check",
    "content marketer",
    "holds a bachelor",
    "contributor",
    "specialties include",
)


def normalize_text(text):
    return re.sub(r"\s+", " ", (text or "")).strip()


def is_noise_chunk(text):
    t = normalize_text(text).lower()
    if not t:
        return True
    if any(t.startswith(prefix) for prefix in NOISE_PREFIXES):
        return True
    if any(token in t for token in NOISE_TOKENS):
        return True
    # Suppress short lines that are mostly bio-like and low-signal.
    if len(t.split()) < 35 and (" is an " in t or " has been " in t):
        return True
    # Detect biography lead-ins that can be prepended to useful chunks.
    bio_lead = re.match(r"^[a-z][a-z.\-']+(?:\s+[a-z][a-z.\-']+){1,5}\s+is\s+an?\s+", t) is not None
    if bio_lead and any(token in t for token in (
        "years of experience", "fact-checker", "editor", "analyst", "broker", "writer", "is a cpa", "has a degree"
    )):
        return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(description="Upload chunk embeddings to Pinecone.")
    parser.add_argument("--index", default=None, help="Pinecone index name")
    parser.add_argument("--chunk-files", nargs="+", default=DEFAULT_CHUNK_FILES, help="Chunk JSON files")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Upsert batch size")
    parser.add_argument("--namespace", default="", help="Optional namespace override")
    parser.add_argument("--no-filter-noise", action="store_true", help="Disable noise filtering before upload")
    parser.add_argument("--clear-namespace", action="store_true", help="Delete all vectors in namespace before upload")
    parser.add_argument("--dry-run", action="store_true", help="Validate input files without uploading")
    return parser.parse_args()


def parse_environment(value):
    if not value:
        return None, None

    # Legacy format example: us-east1-gcp
    parts = value.split("-")
    if len(parts) >= 3 and parts[-1] in {"aws", "gcp", "azure"}:
        cloud = parts[-1]
        region = "-".join(parts[:-1])
        return cloud, region

    return None, None


def load_chunks(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {file_path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        return data, 0

    dim = len(data[0].get("embedding", []))
    return data, dim


def create_index_if_missing(pc, index_name, dimension):
    existing = pc.list_indexes().names()
    if index_name in existing:
        return

    cloud = os.getenv("PINECONE_CLOUD")
    region = os.getenv("PINECONE_REGION")

    if not cloud or not region:
        env_value = os.getenv("PINECONE_ENVIRONMENT", "")
        parsed_cloud, parsed_region = parse_environment(env_value)
        cloud = cloud or parsed_cloud or "gcp"
        region = region or parsed_region or "us-east1"

    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region),
    )


def upload_chunks(index, chunk_file, batch_size=50, namespace="", filter_noise=True, clear_namespace=False):
    data, dim = load_chunks(chunk_file)
    if not data:
        print(f"Skipped empty file: {chunk_file}")
        return 0, dim

    if filter_noise:
        original_len = len(data)
        data = [item for item in data if not is_noise_chunk(item.get("text", ""))]
        removed = original_len - len(data)
        if removed:
            print(f"Filtered noise chunks from {chunk_file}: removed {removed}, kept {len(data)}")

    if clear_namespace:
        index.delete(delete_all=True, namespace=namespace)
        print(f"Cleared namespace '{namespace}' before upload")

    for i in tqdm(range(0, len(data), batch_size), desc=f"Uploading {chunk_file}"):
        batch = data[i:i + batch_size]
        vectors = []
        for item in batch:
            metadata = dict(item.get("metadata", {}))
            metadata["text"] = item.get("text", "")
            vectors.append({
                "id": item["id"],
                "values": item["embedding"],
                "metadata": metadata,
            })
        index.upsert(vectors=vectors, namespace=namespace)

    print(f"Uploaded {len(data)} vectors from {chunk_file}")
    return len(data), dim


def main():
    load_dotenv()
    args = parse_args()
    index_name = args.index or os.getenv("PINECONE_INDEX") or DEFAULT_INDEX_NAME

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key and not args.dry_run:
        raise ValueError("PINECONE_API_KEY is missing. Set it in environment or .env file.")

    # Validate all files and ensure embedding dimension consistency.
    dims = set()
    file_sizes = {}
    for chunk_file in args.chunk_files:
        data, dim = load_chunks(chunk_file)
        if data:
            dims.add(dim)
        file_sizes[chunk_file] = len(data)

    if len(dims) > 1:
        raise ValueError(f"Embedding dimension mismatch across files: {sorted(dims)}")

    if args.dry_run:
        print("Dry run complete. File stats:")
        for file_name, count in file_sizes.items():
            print(f"- {file_name}: {count} vectors")
        print(f"- embedding_dim: {next(iter(dims), 0)}")
        return

    pc = Pinecone(api_key=api_key)
    dimension = next(iter(dims), 384)
    create_index_if_missing(pc, index_name, dimension)
    index = pc.Index(index_name)

    total_uploaded = 0
    filter_noise = not args.no_filter_noise

    for chunk_file in args.chunk_files:
        namespace = args.namespace
        if not namespace:
            namespace = Path(chunk_file).stem.replace("_chunks", "")
        uploaded, _ = upload_chunks(
            index,
            chunk_file,
            batch_size=args.batch_size,
            namespace=namespace,
            filter_noise=filter_noise,
            clear_namespace=args.clear_namespace,
        )
        total_uploaded += uploaded

    print(f"Done. Total uploaded vectors: {total_uploaded}")


if __name__ == "__main__":
    main()
