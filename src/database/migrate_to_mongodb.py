import json
from pathlib import Path

from pymongo import UpdateOne

from mongodb import get_database


CHUNKS_PATH = Path("data/processed/policy_chunks.json")
NT_EMBEDDINGS_PATH = Path("data/processed/policy_embeddings.json")
SUMMARIES_PATH = Path("data/processed/policy_heading_summaries.json")
SNT_EMBEDDINGS_PATH = Path(
    "data/processed/policy_summary_embeddings.json"
)
IE_EMBEDDINGS_PATH = Path(
    "data/processed/policy_internal_embeddings.json"
)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find: {path.resolve()}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def main():
    db = get_database()

    chunks_collection = db["chunks"]
    ie_collection = db["internal_embeddings"]

    chunks_data = load_json(CHUNKS_PATH)["chunks"]
    nt_records = load_json(NT_EMBEDDINGS_PATH)["records"]
    summary_records = load_json(SUMMARIES_PATH)["records"]
    snt_records = load_json(SNT_EMBEDDINGS_PATH)["records"]
    ie_records = load_json(IE_EMBEDDINGS_PATH)["records"]

    print(f"Loaded {len(chunks_data)} chunks.")
    print(f"Loaded {len(nt_records)} NT embeddings.")
    print(f"Loaded {len(summary_records)} summaries.")
    print(f"Loaded {len(snt_records)} SNT embeddings.")
    print(f"Loaded {len(ie_records)} IE embeddings.")

    # --------------------------------------------------
    # Build lookups
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Build chunk documents
    # --------------------------------------------------

    chunk_operations = []

    for index, chunk in enumerate(
        chunks_data,
        start=1,
    ):
        chunk_id = f"{index:03d}"

        if chunk_id not in nt_lookup:
            raise KeyError(
                f"Missing NT embedding for chunk {chunk_id}"
            )

        if chunk_id not in summary_lookup:
            raise KeyError(
                f"Missing summary for chunk {chunk_id}"
            )

        if chunk_id not in snt_lookup:
            raise KeyError(
                f"Missing SNT embedding for chunk {chunk_id}"
            )

        document = {
            "chunk_id": chunk_id,
            "chapter": chunk["chapter"],
            "headings": chunk["headings"],
            "section_ids": chunk["section_ids"],
            "content": chunk["content"],
            "summary": summary_lookup[chunk_id]["summary"],
            "nt_embedding": nt_lookup[chunk_id]["embedding"],
            "snt_embedding": snt_lookup[chunk_id]["embedding"],
        }

        chunk_operations.append(
            UpdateOne(
                {"chunk_id": chunk_id},
                {"$set": document},
                upsert=True,
            )
        )

    # --------------------------------------------------
    # Build IE documents
    # --------------------------------------------------

    ie_operations = []

    for record in ie_records:
        document = {
            "internal_id": record["internal_id"],
            "chunk_id": record["chunk_id"],
            "ie_heading": record["ie_heading"],
            "ie_content": record["ie_content"],
            "ie_embedding": record["embedding"],
        }

        ie_operations.append(
            UpdateOne(
                {
                    "internal_id": record["internal_id"]
                },
                {"$set": document},
                upsert=True,
            )
        )

    # --------------------------------------------------
    # Write to MongoDB
    # --------------------------------------------------

    if chunk_operations:
        chunk_result = chunks_collection.bulk_write(
            chunk_operations
        )

        print(
            "Chunks migration complete."
        )
        print(
            f"Matched: {chunk_result.matched_count}"
        )
        print(
            f"Modified: {chunk_result.modified_count}"
        )
        print(
            f"Upserted: {chunk_result.upserted_count}"
        )

    if ie_operations:
        ie_result = ie_collection.bulk_write(
            ie_operations
        )

        print(
            "\nInternal embeddings migration complete."
        )
        print(
            f"Matched: {ie_result.matched_count}"
        )
        print(
            f"Modified: {ie_result.modified_count}"
        )
        print(
            f"Upserted: {ie_result.upserted_count}"
        )

    print(
        "\nMongoDB migration complete."
    )


if __name__ == "__main__":
    main()