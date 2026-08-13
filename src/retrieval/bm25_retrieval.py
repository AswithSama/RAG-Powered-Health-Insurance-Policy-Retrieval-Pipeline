import json
import re

from rank_bm25 import BM25Okapi

from src.config import (
    POLICY_CHUNKS_PATH,
    CHUNK_TOP_K,
)


def load_chunks() -> list[dict]:
    if not POLICY_CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Could not find: {POLICY_CHUNKS_PATH.resolve()}")

    data = json.loads(POLICY_CHUNKS_PATH.read_text(encoding="utf-8"))
    return data["chunks"]


def tokenize(text: str) -> list[str]:
    """
    Simple lowercase tokenizer.
    Keeps alphanumeric words and removes punctuation.
    """
    return re.findall(r"\b\w+\b", text.lower())


def build_bm25_documents(
    chunks: list[dict],
) -> tuple[list[str], list[list[str]]]:
    chunk_ids = []
    tokenized_documents = []

    for index, chunk in enumerate(chunks, start=1):
        chunk_id = f"{index:03d}"

        headings_text = " ".join(chunk["headings"])
        document_text = f"{headings_text}\n{chunk['content']}"

        chunk_ids.append(chunk_id)
        tokenized_documents.append(tokenize(document_text))

    return chunk_ids, tokenized_documents


def retrieve_bm25(
    query: str,
    bm25: BM25Okapi,
    chunk_ids: list[str],
    top_k: int = CHUNK_TOP_K,
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

    return results[:top_k]


def main():
    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks.")

    chunk_ids, tokenized_documents = build_bm25_documents(chunks)

    bm25 = BM25Okapi(tokenized_documents)

    print("BM25 index created.")

    while True:
        query = input("\nEnter query (or 'exit'): ").strip()

        if query.lower() == "exit":
            break

        results = retrieve_bm25(
            query=query,
            bm25=bm25,
            chunk_ids=chunk_ids,
            top_k=CHUNK_TOP_K,
        )

        print("\nBM25 results:\n")

        for rank, result in enumerate(results, start=1):
            print(
                f"{rank}. "
                f"Chunk {result['chunk_id']} | "
                f"BM25: {result['bm25_score']:.4f}"
            )


if __name__ == "__main__":
    main()