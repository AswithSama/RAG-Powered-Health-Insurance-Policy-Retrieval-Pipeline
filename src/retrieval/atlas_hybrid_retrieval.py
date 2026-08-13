import re

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL,
    W_NT,
    W_SNT,
    RRF_K,
    CHUNK_TOP_K,
    IE_TOP_K,
    CHUNKS_COLLECTION,
    INTERNAL_EMBEDDINGS_COLLECTION,
    NT_VECTOR_INDEX,
    SNT_VECTOR_INDEX,
    IE_VECTOR_INDEX,
    NT_VECTOR_FIELD,
    SNT_VECTOR_FIELD,
    IE_VECTOR_FIELD,
    VECTOR_NUM_CANDIDATES,
)

from src.database.mongodb import (
    get_database,
    vector_search_chunks,
    vector_search_internal_embeddings,
)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def load_chunks_from_mongodb(collection) -> list[dict]:
    return list(
        collection.find(
            {},
            {
                "_id": 0,
                "chunk_id": 1,
                "headings": 1,
                "content": 1,
            },
        )
    )


def build_bm25(chunks: list[dict]) -> tuple[BM25Okapi, list[str]]:
    chunk_ids = []
    tokenized_documents = []

    for chunk in chunks:
        headings_text = " ".join(chunk["headings"])
        document_text = f"{headings_text}\n{chunk['content']}"

        chunk_ids.append(chunk["chunk_id"])
        tokenized_documents.append(tokenize(document_text))

    bm25 = BM25Okapi(tokenized_documents)

    return bm25, chunk_ids


def atlas_dense_scores(
    query_embedding: list[float],
    chunks_collection,
) -> list[dict]:

    nt_results = vector_search_chunks(
        collection=chunks_collection,
        query_embedding=query_embedding,
        index_name=NT_VECTOR_INDEX,
        path=NT_VECTOR_FIELD,
        limit=38,
        num_candidates=VECTOR_NUM_CANDIDATES,
    )

    snt_results = vector_search_chunks(
        collection=chunks_collection,
        query_embedding=query_embedding,
        index_name=SNT_VECTOR_INDEX,
        path=SNT_VECTOR_FIELD,
        limit=38,
        num_candidates=VECTOR_NUM_CANDIDATES,
    )

    nt_lookup = {
        item["chunk_id"]: item["score"]
        for item in nt_results
    }

    snt_lookup = {
        item["chunk_id"]: item["score"]
        for item in snt_results
    }

    chunk_ids = set(nt_lookup) & set(snt_lookup)

    results = []

    for chunk_id in chunk_ids:
        nt_score = nt_lookup[chunk_id]
        snt_score = snt_lookup[chunk_id]

        dense_score = W_NT * nt_score + W_SNT * snt_score

        results.append(
            {
                "chunk_id": chunk_id,
                "nt_score": nt_score,
                "snt_score": snt_score,
                "dense_score": dense_score,
            }
        )

    results.sort(key=lambda x: x["dense_score"], reverse=True)

    for rank, result in enumerate(results, start=1):
        result["dense_rank"] = rank

    return results


def bm25_scores(
    query: str,
    bm25: BM25Okapi,
    chunk_ids: list[str],
) -> list[dict]:

    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    results = []

    for chunk_id, score in zip(chunk_ids, scores):
        results.append(
            {
                "chunk_id": chunk_id,
                "bm25_score": float(score),
            }
        )

    results.sort(key=lambda x: x["bm25_score"], reverse=True)

    for rank, result in enumerate(results, start=1):
        result["bm25_rank"] = rank

    return results


def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
) -> list[dict]:

    dense_lookup = {
        item["chunk_id"]: item
        for item in dense_results
    }

    bm25_lookup = {
        item["chunk_id"]: item
        for item in bm25_results
    }

    chunk_ids = set(dense_lookup) & set(bm25_lookup)

    fused_results = []

    for chunk_id in chunk_ids:
        dense_item = dense_lookup[chunk_id]
        bm25_item = bm25_lookup[chunk_id]

        dense_rank = dense_item["dense_rank"]
        bm25_rank = bm25_item["bm25_rank"]

        rrf_score = (
            1 / (RRF_K + dense_rank)
            + 1 / (RRF_K + bm25_rank)
        )

        fused_results.append(
            {
                "chunk_id": chunk_id,
                "nt_score": dense_item["nt_score"],
                "snt_score": dense_item["snt_score"],
                "dense_score": dense_item["dense_score"],
                "dense_rank": dense_rank,
                "bm25_score": bm25_item["bm25_score"],
                "bm25_rank": bm25_rank,
                "rrf_score": rrf_score,
            }
        )

    fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)

    return fused_results


def atlas_hybrid_retrieve(
    query: str,
    model: SentenceTransformer,
    chunks_collection,
    ie_collection,
    bm25: BM25Okapi,
    bm25_chunk_ids: list[str],
) -> list[dict]:

    # Query embedding
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    # Atlas NT + SNT retrieval
    dense_results = atlas_dense_scores(
        query_embedding=query_embedding,
        chunks_collection=chunks_collection,
    )

    # BM25 retrieval
    lexical_results = bm25_scores(
        query=query,
        bm25=bm25,
        chunk_ids=bm25_chunk_ids,
    )

    # RRF fusion
    fused_results = reciprocal_rank_fusion(
        dense_results,
        lexical_results,
    )

    top_chunks = fused_results[:CHUNK_TOP_K]

    selected_chunk_ids = [
        result["chunk_id"]
        for result in top_chunks
    ]

    # Atlas IE vector search
    ie_results = vector_search_internal_embeddings(
        collection=ie_collection,
        query_embedding=query_embedding,
        selected_chunk_ids=selected_chunk_ids,
        index_name=IE_VECTOR_INDEX,
        path=IE_VECTOR_FIELD,
        limit=IE_TOP_K,
        num_candidates=VECTOR_NUM_CANDIDATES,
    )

    # Rename Atlas score for compatibility
    for result in ie_results:
        result["ie_score"] = result.pop("score")

    return ie_results


def main():
    db = get_database()

    chunks_collection = db[CHUNKS_COLLECTION]
    ie_collection = db[INTERNAL_EMBEDDINGS_COLLECTION]

    chunks = load_chunks_from_mongodb(chunks_collection)

    print(f"Loaded {len(chunks)} chunks from MongoDB Atlas.")

    bm25, bm25_chunk_ids = build_bm25(chunks)

    print(f"Loading embedding model: {EMBEDDING_MODEL}")

    model = SentenceTransformer(EMBEDDING_MODEL)

    while True:
        query = input("\nEnter query (or 'exit'): ").strip()

        if query.lower() == "exit":
            break

        ie_results = atlas_hybrid_retrieve(
            query=query,
            model=model,
            chunks_collection=chunks_collection,
            ie_collection=ie_collection,
            bm25=bm25,
            bm25_chunk_ids=bm25_chunk_ids,
        )

        print("\nAtlas IE results:\n")

        for rank, result in enumerate(ie_results, start=1):
            print(
                f"{rank}. "
                f"{result['internal_id']} | "
                f"Chunk {result['chunk_id']} | "
                f"{result['ie_heading']} | "
                f"Score: {result['ie_score']:.4f}"
            )


if __name__ == "__main__":
    main()