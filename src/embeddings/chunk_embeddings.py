from __future__ import annotations

import json

from sentence_transformers import SentenceTransformer

from src.config import (
    POLICY_CHUNKS_PATH,
    CHUNK_EMBEDDINGS_PATH,
    EMBEDDING_MODEL,
)


def load_chunks() -> list[dict]:
    if not POLICY_CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find: {POLICY_CHUNKS_PATH.resolve()}"
        )

    data = json.loads(POLICY_CHUNKS_PATH.read_text(encoding="utf-8"))

    return data["chunks"]


def main():
    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks.")
    print(f"Loading embedding model: {EMBEDDING_MODEL}")

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [chunk["content"] for chunk in chunks]

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    records = []

    for chunk, embedding in zip(chunks, embeddings):
        records.append(
            {
                "chunk_id": chunk["chunk_id"],
                "chapter": chunk["chapter"],
                "headings": chunk["headings"],
                "start_pdf_page": chunk["start_pdf_page"],
                "end_pdf_page": chunk["end_pdf_page"],
                "content": chunk["content"],
                "embedding": embedding.tolist(),
            }
        )

    output = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": len(embeddings[0]),
        "total_chunks": len(records),
        "records": records,
    }

    CHUNK_EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    CHUNK_EMBEDDINGS_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nEmbedding creation complete.")
    print(f"Total embeddings: {len(records)}")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"Saved embeddings to: {CHUNK_EMBEDDINGS_PATH.resolve()}")


if __name__ == "__main__":
    main()