import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


SUMMARIES_PATH = Path("data/processed/policy_heading_summaries.json")
OUTPUT_PATH = Path("data/processed/policy_summary_embeddings.json")

MODEL_NAME = "BAAI/bge-base-en-v1.5"


def load_summaries() -> list[dict]:
    if not SUMMARIES_PATH.exists():
        raise FileNotFoundError(f"Could not find: {SUMMARIES_PATH.resolve()}")

    data = json.loads(SUMMARIES_PATH.read_text(encoding="utf-8"))
    return data["records"]


def main():
    records = load_summaries()

    print(f"Loaded {len(records)} chunk summaries.")
    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    # One merged chunk summary -> one SNT embedding
    texts = [record["summary"] for record in records]

    print("Creating summary embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    embedding_records = []

    for record, embedding in zip(records, embeddings):
        embedding_records.append(
            {
                "chunk_id": record["chunk_id"],
                "summary": record["summary"],
                "embedding": embedding.tolist(),
            }
        )

    output = {
        "embedding_model": MODEL_NAME,
        "embedding_dimension": len(embeddings[0]),
        "total_summary_embeddings": len(embedding_records),
        "records": embedding_records,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSummary embedding creation complete.")
    print(f"Total summary embeddings: {len(embedding_records)}")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"Saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()