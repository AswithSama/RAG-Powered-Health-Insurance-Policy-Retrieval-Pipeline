import json

from sentence_transformers import SentenceTransformer

from src.config import (
    POLICY_SUMMARIES_PATH,
    SUMMARY_EMBEDDINGS_PATH,
    EMBEDDING_MODEL,
)


def load_summaries() -> list[dict]:
    if not POLICY_SUMMARIES_PATH.exists():
        raise FileNotFoundError(
            f"Could not find: {POLICY_SUMMARIES_PATH.resolve()}"
        )

    data = json.loads(POLICY_SUMMARIES_PATH.read_text(encoding="utf-8"))
    return data["records"]


def main():
    records = load_summaries()

    print(f"Loaded {len(records)} chunk summaries.")
    print(f"Loading embedding model: {EMBEDDING_MODEL}")

    model = SentenceTransformer(EMBEDDING_MODEL)

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
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": len(embeddings[0]),
        "total_summary_embeddings": len(embedding_records),
        "records": embedding_records,
    }

    SUMMARY_EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    SUMMARY_EMBEDDINGS_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSummary embedding creation complete.")
    print(f"Total summary embeddings: {len(embedding_records)}")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"Saved to: {SUMMARY_EMBEDDINGS_PATH.resolve()}")


if __name__ == "__main__":
    main()