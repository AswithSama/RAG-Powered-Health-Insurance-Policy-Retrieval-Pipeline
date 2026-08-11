from sentence_transformers import CrossEncoder, SentenceTransformer

from hybrid_retrieval import (
    NT_EMBEDDINGS_PATH,
    SNT_EMBEDDINGS_PATH,
    IE_EMBEDDINGS_PATH,
    MODEL_NAME,
    load_records,
    load_chunks,
    build_embedding_lookup,
    build_bm25,
    hybrid_retrieve,
)


CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
CROSS_ENCODER_TOP_K = 5
CROSS_ENCODER_THRESHOLD = -5.0


def rerank_with_cross_encoder(
    query: str,
    ie_results: list[dict],
    cross_encoder: CrossEncoder,
    top_k: int = CROSS_ENCODER_TOP_K,
) -> list[dict]:

    # Query-passage pairs
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
        if result["cross_encoder_score"]
        >= CROSS_ENCODER_THRESHOLD
    ]

    return filtered_results[:top_k]


def main():

    # --------------------------------------------------
    # Load retrieval data
    # --------------------------------------------------

    nt_records = load_records(NT_EMBEDDINGS_PATH)
    snt_records = load_records(SNT_EMBEDDINGS_PATH)
    ie_records = load_records(IE_EMBEDDINGS_PATH)

    chunks = load_chunks()

    print(f"Loaded {len(nt_records)} NT embeddings.")
    print(f"Loaded {len(snt_records)} SNT embeddings.")
    print(f"Loaded {len(ie_records)} IE embeddings.")

    # --------------------------------------------------
    # Build retrieval structures
    # --------------------------------------------------

    nt_lookup = build_embedding_lookup(nt_records)
    snt_lookup = build_embedding_lookup(snt_records)

    if set(nt_lookup) != set(snt_lookup):
        raise ValueError("NT and SNT chunk IDs do not match.")

    bm25, bm25_chunk_ids = build_bm25(chunks)

    # --------------------------------------------------
    # Load models
    # --------------------------------------------------

    print(f"Loading embedding model: {MODEL_NAME}")

    embedding_model = SentenceTransformer(MODEL_NAME)

    print(f"Loading CrossEncoder: {CROSS_ENCODER_MODEL}")

    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    # --------------------------------------------------
    # Query loop
    # --------------------------------------------------

    while True:
        query = input("\nEnter query (or 'exit'): ").strip()

        if query.lower() == "exit":
            break

        # --------------------------------------------------
        # Run hybrid retrieval
        # --------------------------------------------------

        ie_results = hybrid_retrieve(
            query=query,
            model=embedding_model,
            nt_lookup=nt_lookup,
            snt_lookup=snt_lookup,
            bm25=bm25,
            bm25_chunk_ids=bm25_chunk_ids,
            ie_records=ie_records,
        )

        print("\nIE cosine candidates:\n")

        for rank, result in enumerate(ie_results, start=1):
            print(
                f"{rank}. "
                f"{result['internal_id']} | "
                f"{result['ie_heading']} | "
                f"Cosine: {result['ie_score']:.4f}"
            )

        # --------------------------------------------------
        # CrossEncoder reranking
        # --------------------------------------------------

        reranked_results = rerank_with_cross_encoder(
            query=query,
            ie_results=ie_results,
            cross_encoder=cross_encoder,
        )

        print("\nCrossEncoder results:\n")

        for rank, result in enumerate(reranked_results, start=1):
            print(
                f"{rank}. "
                f"{result['internal_id']} | "
                f"Chunk {result['chunk_id']} | "
                f"{result['ie_heading']} | "
                f"IE cosine: {result['ie_score']:.4f} | "
                f"CrossEncoder: {result['cross_encoder_score']:.4f}"
            )

            #eprint(f"\n{result['ie_content']}\n")


if __name__ == "__main__":
    main()