import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


NT_EMBEDDINGS_PATH = Path("data/processed/policy_embeddings.json")
SNT_EMBEDDINGS_PATH = Path("data/processed/policy_summary_embeddings.json")

MODEL_NAME = "BAAI/bge-base-en-v1.5"

W_NT = 0.5
W_SNT = 0.5

TOP_K = 10


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find: {path.resolve()}")

    data = json.loads(path.read_text(encoding="utf-8"))
    return data["records"]


def build_embedding_lookup(records: list[dict]) -> dict[str, np.ndarray]:
    return {
        record["chunk_id"]: np.array(record["embedding"], dtype=np.float32)
        for record in records
    }


def retrieve(
    query: str,
    model: SentenceTransformer,
    nt_records: list[dict],
    snt_records: list[dict],
    top_k: int = TOP_K,
) -> list[dict]:

    # --------------------------------------------------
    # Query embedding
    # --------------------------------------------------

    query_embedding = model.encode(query, normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype=np.float32)

    # --------------------------------------------------
    # Build lookup by chunk_id
    # --------------------------------------------------

    nt_lookup = build_embedding_lookup(nt_records)
    snt_lookup = build_embedding_lookup(snt_records)

    # --------------------------------------------------
    # Validate alignment
    # --------------------------------------------------

    nt_chunk_ids = set(nt_lookup)
    snt_chunk_ids = set(snt_lookup)

    if nt_chunk_ids != snt_chunk_ids:
        raise ValueError("NT and SNT chunk IDs do not match.")

    # --------------------------------------------------
    # Score each chunk
    # --------------------------------------------------

    results = []

    for chunk_id in nt_lookup:
        nt_embedding = nt_lookup[chunk_id]
        snt_embedding = snt_lookup[chunk_id]

        # Since all vectors are normalized,
        # dot product == cosine similarity.
        nt_score = float(np.dot(query_embedding, nt_embedding))
        snt_score = float(np.dot(query_embedding, snt_embedding))

        dense_score = W_NT * nt_score + W_SNT * snt_score

        results.append(
            {
                "chunk_id": chunk_id,
                "nt_score": nt_score,
                "snt_score": snt_score,
                "dense_score": dense_score,
            }
        )

    # --------------------------------------------------
    # Rank
    # --------------------------------------------------

    results.sort(key=lambda x: x["dense_score"], reverse=True)

    return results[:top_k]


def main():
    nt_records = load_records(NT_EMBEDDINGS_PATH)
    snt_records = load_records(SNT_EMBEDDINGS_PATH)

    print(f"Loaded {len(nt_records)} NT embeddings.")
    print(f"Loaded {len(snt_records)} SNT embeddings.")
    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    while True:
        query = input("\nEnter query (or 'exit'): ").strip()

        if query.lower() == "exit":
            break

        results = retrieve(
            query=query,
            model=model,
            nt_records=nt_records,
            snt_records=snt_records,
            top_k=TOP_K,
        )

        print("\nTop results:\n")

        for rank, result in enumerate(results, start=1):
            print(
                f"{rank}. "
                f"Chunk {result['chunk_id']} | "
                f"NT: {result['nt_score']:.4f} | "
                f"SNT: {result['snt_score']:.4f} | "
                f"Dense: {result['dense_score']:.4f}"
            )


if __name__ == "__main__":
    main()