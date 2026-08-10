import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDINGS_PATH = Path("policy_embeddings.json")

MODEL_NAME = "BAAI/bge-base-en-v1.5"

QUERY = "Does the plan cover ambulance transportation?"

TOP_K = 5


def load_records():
    data = json.loads(
        EMBEDDINGS_PATH.read_text(encoding="utf-8")
    )

    return data["records"]


def cosine_similarity(query_vector, document_vector):
    """
    Cosine similarity between two vectors.
    """

    query_vector = np.asarray(query_vector)
    document_vector = np.asarray(document_vector)

    return np.dot(query_vector, document_vector) / (
        np.linalg.norm(query_vector)
        * np.linalg.norm(document_vector)
    )


def main():
    records = load_records()

    print(f"Loaded {len(records)} embedded chunks.")
    print(f"\nQuery: {QUERY}\n")

    model = SentenceTransformer(MODEL_NAME)

    # IMPORTANT:
    # Use the same embedding model and normalization
    # used when creating document embeddings.
    query_embedding = model.encode(
        QUERY,
        normalize_embeddings=True,
    )

    results = []

    for record in records:

        score = cosine_similarity(
            query_embedding,
            record["embedding"],
        )

        results.append(
            {
                "score": float(score),
                "chunk_id": record["chunk_id"],
                "chapter": record["chapter"],
                "headings": record["headings"],
                "content": record["content"],
            }
        )

    # Highest similarity first
    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    print("=" * 100)
    print(f"TOP {TOP_K} RETRIEVAL RESULTS")
    print("=" * 100)

    for rank, result in enumerate(
        results[:TOP_K],
        start=1,
    ):
        print(f"\nRank: {rank}")
        print(f"Score: {result['score']:.4f}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Chapter: {result['chapter']}")

        print(
            "Headings:",
            ", ".join(result["headings"])
        )

        print("-" * 100)

        # Only show a preview for now
        print(result["content"][:1000])

        print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
