import json
from pathlib import Path

from pymongo import UpdateOne

from src.config import (
    POLICY_CHUNKS_PATH,
    CHUNK_EMBEDDINGS_PATH,
    POLICY_SUMMARIES_PATH,
    SUMMARY_EMBEDDINGS_PATH,
    INTERNAL_EMBEDDINGS_PATH,
    CHUNKS_COLLECTION,
    INTERNAL_EMBEDDINGS_COLLECTION,
    NT_VECTOR_FIELD,
    SNT_VECTOR_FIELD,
    IE_VECTOR_FIELD,
)

from src.database.mongodb import get_database


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Could not find: {path.resolve()}")

    return json.loads(path.read_text(encoding="utf-8"))


def main():
    db = get_database()

    chunks_collection = db[CHUNKS_COLLECTION]
    ie_collection = db[INTERNAL_EMBEDDINGS_COLLECTION]

    chunks_data = load_json(POLICY_CHUNKS_PATH)["chunks"]
    nt_records = load_json(CHUNK_EMBEDDINGS_PATH)["records"]
    summary_records = load_json(POLICY_SUMMARIES_PATH)["records"]
    snt_records = load_json(SUMMARY_EMBEDDINGS_PATH)["records"]
    ie_records = load_json(INTERNAL_EMBEDDINGS_PATH)["records"]

    print(f"Loaded {len(chunks_data)} chunks.")
    print(f"Loaded {len(nt_records)} NT embeddings.")
    print(f"Loaded {len(summary_records)} summaries.")
    print(f"Loaded {len(snt_records)} SNT embeddings.")
    print(f"Loaded {len(ie_records)} IE embeddings.")

    # Build lookups
    nt_lookup = {
        record["chunk_id"]: record
        for record in nt_records
    }

    summary_lookup = {
        record["chunk_id"]: record
        for record in summary_records
    }

    snt_lookup = {
        record["chunk_id"]: record
        for record in snt_records
    }

    # Build chunk documents
    chunk_operations = []

    for index, chunk in enumerate(chunks_data, start=1):
        chunk_id = f"{index:03d}"

        if chunk_id not in nt_lookup:
            raise KeyError(f"Missing NT embedding for chunk {chunk_id}")

        if chunk_id not in summary_lookup:
            raise KeyError(f"Missing summary for chunk {chunk_id}")

        if chunk_id not in snt_lookup:
            raise KeyError(f"Missing SNT embedding for chunk {chunk_id}")

        document = {
            "chunk_id": chunk_id,
            "chapter": chunk["chapter"],
            "headings": chunk["headings"],
            "section_ids": chunk["section_ids"],
            "content": chunk["content"],
            "summary": summary_lookup[chunk_id]["summary"],
            NT_VECTOR_FIELD: nt_lookup[chunk_id]["embedding"],
            SNT_VECTOR_FIELD: snt_lookup[chunk_id]["embedding"],
        }

        chunk_operations.append(
            UpdateOne(
                {"chunk_id": chunk_id},
                {"$set": document},
                upsert=True,
            )
        )

    # Build IE documents
    ie_operations = []

    for record in ie_records:
        document = {
            "internal_id": record["internal_id"],
            "chunk_id": record["chunk_id"],
            "ie_heading": record["ie_heading"],
            "ie_content": record["ie_content"],
            IE_VECTOR_FIELD: record["embedding"],
        }

        ie_operations.append(
            UpdateOne(
                {"internal_id": record["internal_id"]},
                {"$set": document},
                upsert=True,
            )
        )

    # Write chunks
    if chunk_operations:
        chunk_result = chunks_collection.bulk_write(chunk_operations)

        print("\nChunks migration complete.")
        print(f"Matched: {chunk_result.matched_count}")
        print(f"Modified: {chunk_result.modified_count}")
        print(f"Upserted: {chunk_result.upserted_count}")

    # Write internal embeddings
    if ie_operations:
        ie_result = ie_collection.bulk_write(ie_operations)

        print("\nInternal embeddings migration complete.")
        print(f"Matched: {ie_result.matched_count}")
        print(f"Modified: {ie_result.modified_count}")
        print(f"Upserted: {ie_result.upserted_count}")

    print("\nMongoDB migration complete.")


if __name__ == "__main__":
    main()