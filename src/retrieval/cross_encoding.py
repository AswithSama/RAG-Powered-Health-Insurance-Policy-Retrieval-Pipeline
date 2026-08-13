from sentence_transformers import CrossEncoder, SentenceTransformer

from src.config import (
    EMBEDDING_MODEL,
    CROSS_ENCODER_MODEL,
    CROSS_ENCODER_TOP_K,
    CROSS_ENCODER_THRESHOLD,
    CHUNKS_COLLECTION,
    INTERNAL_EMBEDDINGS_COLLECTION,
)

from src.database.mongodb import get_database

from src.retrieval.atlas_hybrid_retrieval import (
    load_chunks_from_mongodb,
    build_bm25,
    atlas_hybrid_retrieve,
)


def rerank_with_cross_encoder(
    query: str,
    ie_results: list[dict],
    cross_encoder: CrossEncoder,
    top_k: int = CROSS_ENCODER_TOP_K,
) -> list[dict]:
    pairs = [
        (query, result["ie_content"])
        for result in ie_results
    ]

    scores = cross_encoder.predict(pairs)

    reranked_results = []

    for result, score in zip(ie_results, scores):
        reranked_results.append(
            {
                **result,
                "cross_encoder_score": float(score),
            }
        )

    reranked_results.sort(
        key=lambda x: x["cross_encoder_score"],
        reverse=True,
    )

    filtered_results = [
        result
        for result in reranked_results
        if result["cross_encoder_score"] >= CROSS_ENCODER_THRESHOLD
    ]

    return filtered_results[:top_k]


def main():
    # MongoDB
    db = get_database()

    chunks_collection = db[CHUNKS_COLLECTION]
    ie_collection = db[INTERNAL_EMBEDDINGS_COLLECTION]

    # Load chunks for BM25
    chunks = load_chunks_from_mongodb(chunks_collection)

    print(f"Loaded {len(chunks)} chunks from MongoDB Atlas.")

    bm25, bm25_chunk_ids = build_bm25(chunks)

    # Load models
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Loading CrossEncoder: {CROSS_ENCODER_MODEL}")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    # Query loop
    while True:
        query = input("\nEnter query (or 'exit'): ").strip()

        if query.lower() == "exit":
            break

        # Atlas hybrid retrieval
        ie_results = atlas_hybrid_retrieve(
            query=query,
            model=embedding_model,
            chunks_collection=chunks_collection,
            ie_collection=ie_collection,
            bm25=bm25,
            bm25_chunk_ids=bm25_chunk_ids,
        )

        print("\nAtlas IE candidates:\n")

        for rank, result in enumerate(ie_results, start=1):
            print(
                f"{rank}. "
                f"{result['internal_id']} | "
                f"{result['ie_heading']} | "
                f"Atlas score: {result['ie_score']:.4f}"
            )

        # CrossEncoder reranking
        reranked_results = rerank_with_cross_encoder(
            query=query,
            ie_results=ie_results,
            cross_encoder=cross_encoder,
        )

        print("\nCrossEncoder results:\n")

        if not reranked_results:
            print("No IE passages passed the CrossEncoder threshold.")
            continue

        for rank, result in enumerate(reranked_results, start=1):
            print(
                f"{rank}. "
                f"{result['internal_id']} | "
                f"Chunk {result['chunk_id']} | "
                f"{result['ie_heading']} | "
                f"Atlas score: {result['ie_score']:.4f} | "
                f"CrossEncoder: {result['cross_encoder_score']:.4f}"
            )


if __name__ == "__main__":
    main()