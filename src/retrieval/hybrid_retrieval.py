import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


NT_EMBEDDINGS_PATH = Path("data/processed/policy_embeddings.json")
SNT_EMBEDDINGS_PATH = Path("data/processed/policy_summary_embeddings.json")
CHUNKS_PATH = Path("data/processed/policy_chunks.json")
IE_EMBEDDINGS_PATH = Path("data/processed/policy_internal_embeddings.json")

IE_TOP_K = 20

MODEL_NAME = "BAAI/bge-base-en-v1.5"

W_NT = 0.5
W_SNT = 0.5

RRF_K = 60
TOP_K = 10


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find: {path.resolve()}")

    data = json.loads(path.read_text(encoding="utf-8"))
    return data["records"]


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Could not find: {CHUNKS_PATH.resolve()}")

    data = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return data["chunks"]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def build_embedding_lookup(records: list[dict]) -> dict[str, np.ndarray]:
    return {
        record["chunk_id"]: np.array(record["embedding"], dtype=np.float32)
        for record in records
    }


def build_bm25(chunks: list[dict]) -> tuple[BM25Okapi, list[str]]:
    chunk_ids = []
    tokenized_documents = []

    for index, chunk in enumerate(chunks, start=1):
        chunk_id = f"{index:03d}"

        headings_text = " ".join(chunk["headings"])
        document_text = f"{headings_text}\n{chunk['content']}"

        chunk_ids.append(chunk_id)
        tokenized_documents.append(tokenize(document_text))

    bm25 = BM25Okapi(tokenized_documents)

    return bm25, chunk_ids


def dense_scores(
    query_embedding: np.ndarray,
    nt_lookup: dict[str, np.ndarray],
    snt_lookup: dict[str, np.ndarray],
) -> list[dict]:
    results = []

    for chunk_id in nt_lookup:
        nt_score = float(np.dot(query_embedding, nt_lookup[chunk_id]))
        snt_score = float(np.dot(query_embedding, snt_lookup[chunk_id]))

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

    # Add dense rank
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

    # Add BM25 rank
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

    chunk_ids = set(dense_lookup) | set(bm25_lookup)

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


def retrieve(
    query: str,
    model: SentenceTransformer,
    nt_lookup: dict[str, np.ndarray],
    snt_lookup: dict[str, np.ndarray],
    bm25: BM25Okapi,
    bm25_chunk_ids: list[str],
    top_k: int = TOP_K,
) -> tuple[list[dict], np.ndarray]:
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    query_embedding = np.array(query_embedding, dtype=np.float32)

    dense_results = dense_scores(
        query_embedding,
        nt_lookup,
        snt_lookup,
    )

    lexical_results = bm25_scores(
        query,
        bm25,
        bm25_chunk_ids,
    )

    fused_results = reciprocal_rank_fusion(
        dense_results,
        lexical_results,
    )

    return fused_results[:top_k], query_embedding

def retrieve_internal_embeddings(
    query_embedding: np.ndarray,
    selected_chunk_ids: list[str],
    ie_records: list[dict],
    top_k: int = IE_TOP_K,
) -> list[dict]:

    selected_chunk_ids = set(selected_chunk_ids)

    results = []

    for record in ie_records:

        # Ignore IEs whose parent chunk was not selected
        if record["chunk_id"] not in selected_chunk_ids:
            continue

        ie_embedding = np.array(
            record["embedding"],
            dtype=np.float32,
        )

        # Query and IE embeddings are normalized,
        # so dot product == cosine similarity.
        ie_score = float(
            np.dot(
                query_embedding,
                ie_embedding,
            )
        )

        results.append(
            {
                "internal_id": record["internal_id"],
                "chunk_id": record["chunk_id"],
                "ie_heading": record["ie_heading"],
                "ie_content": record["ie_content"],
                "ie_score": ie_score,
            }
        )

    results.sort(
        key=lambda x: x["ie_score"],
        reverse=True,
    )

    return results[:top_k]

def hybrid_retrieve(
    query: str,
    model: SentenceTransformer,
    nt_lookup: dict[str, np.ndarray],
    snt_lookup: dict[str, np.ndarray],
    bm25: BM25Okapi,
    bm25_chunk_ids: list[str],
    ie_records: list[dict],
) -> list[dict]:

    # --------------------------------------------------
    # Layer 1: Chunk retrieval
    # --------------------------------------------------

    chunk_results, query_embedding = retrieve(
        query=query,
        model=model,
        nt_lookup=nt_lookup,
        snt_lookup=snt_lookup,
        bm25=bm25,
        bm25_chunk_ids=bm25_chunk_ids,
        top_k=TOP_K,
    )

    selected_chunk_ids = [
        result["chunk_id"]
        for result in chunk_results
    ]

    # --------------------------------------------------
    # Layer 2: IE retrieval
    # --------------------------------------------------

    ie_results = retrieve_internal_embeddings(
        query_embedding=query_embedding,
        selected_chunk_ids=selected_chunk_ids,
        ie_records=ie_records,
        top_k=IE_TOP_K,
    )

    return ie_results

def main():
    nt_records = load_records(NT_EMBEDDINGS_PATH)
    snt_records = load_records(SNT_EMBEDDINGS_PATH)
    ie_records = load_records(IE_EMBEDDINGS_PATH)

    chunks = load_chunks()

    print(f"Loaded {len(nt_records)} NT embeddings.")
    print(f"Loaded {len(snt_records)} SNT embeddings.")
    print(f"Loaded {len(chunks)} chunks.")
    print(f"Loaded {len(ie_records)} IE embeddings.")

    nt_lookup = build_embedding_lookup(nt_records)
    snt_lookup = build_embedding_lookup(snt_records)

    if set(nt_lookup) != set(snt_lookup):
        raise ValueError(
            "NT and SNT chunk IDs do not match."
        )

    bm25, bm25_chunk_ids = build_bm25(chunks)

    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    while True:
        query = input(
            "\nEnter query (or 'exit'): "
        ).strip()

        if query.lower() == "exit":
            break

        # --------------------------------------------------
        # First retrieval layer:
        # NT + SNT + BM25 + RRF
        # --------------------------------------------------

        chunk_results, query_embedding = retrieve(
            query=query,
            model=model,
            nt_lookup=nt_lookup,
            snt_lookup=snt_lookup,
            bm25=bm25,
            bm25_chunk_ids=bm25_chunk_ids,
            top_k=TOP_K,
        )

        print("\nHybrid RRF chunk results:\n")

        for rank, result in enumerate(
            chunk_results,
            start=1,
        ):
            print(
                f"{rank}. "
                f"Chunk {result['chunk_id']} | "
                f"Dense: {result['dense_score']:.4f} "
                f"(rank {result['dense_rank']}) | "
                f"BM25: {result['bm25_score']:.4f} "
                f"(rank {result['bm25_rank']}) | "
                f"RRF: {result['rrf_score']:.6f}"
            )

        # --------------------------------------------------
        # Get selected top-10 chunk IDs
        # --------------------------------------------------

        selected_chunk_ids = [
            result["chunk_id"]
            for result in chunk_results
        ]

        # --------------------------------------------------
        # Second retrieval layer:
        # cosine(query, IE)
        # --------------------------------------------------

        ie_results = retrieve_internal_embeddings(
            query_embedding=query_embedding,
            selected_chunk_ids=selected_chunk_ids,
            ie_records=ie_records,
            top_k=IE_TOP_K,
        )

        print("\nInternal embedding results:\n")

        for rank, result in enumerate(
            ie_results,
            start=1,
        ):
            print(
                f"{rank}. "
                f"{result['internal_id']} | "
                f"Chunk {result['chunk_id']} | "
                f"{result['ie_heading']} | "
                f"Cosine: {result['ie_score']:.4f}"
            )



if __name__ == "__main__":
    main()