import os

from dotenv import load_dotenv
from pymongo import MongoClient


load_dotenv()


MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME","policy_rag",)


def get_client() -> MongoClient:
    if not MONGODB_URI:
        raise ValueError(
            "MONGODB_URI is not set in the environment."
        )

    return MongoClient(MONGODB_URI)


def get_database():
    client = get_client()

    return client[MONGODB_DB_NAME]


def test_connection() -> None:
    client = get_client()

    client.admin.command("ping")

    print("MongoDB Atlas connection successful.")

def vector_search_chunks(
    collection,
    query_embedding: list[float],
    index_name: str,
    path: str,
    limit: int = 38,
    num_candidates: int = 100,
) -> list[dict]:

    pipeline = [
        {
            "$vectorSearch": {
                "index": index_name,
                "path": path,
                "queryVector": query_embedding,
                "numCandidates": num_candidates,
                "limit": limit,
            }
        },
        {
            "$project": {
                "_id": 0,
                "chunk_id": 1,
                "score": {
                    "$meta": "vectorSearchScore"
                },
            }
        },
    ]

    return list(collection.aggregate(pipeline))

def vector_search_internal_embeddings(
    collection,
    query_embedding: list[float],
    selected_chunk_ids: list[str],
    limit: int = 20,
    num_candidates: int = 100,
) -> list[dict]:

    pipeline = [
        {
            "$vectorSearch": {
                "index": "ie_vector_index",
                "path": "ie_embedding",
                "queryVector": query_embedding,
                "numCandidates": num_candidates,
                "limit": limit,
                "filter": {
                    "chunk_id": {
                        "$in": selected_chunk_ids
                    }
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "internal_id": 1,
                "chunk_id": 1,
                "ie_heading": 1,
                "ie_content": 1,
                "score": {
                    "$meta": "vectorSearchScore"
                },
            }
        },
    ]

    return list(collection.aggregate(pipeline))

if __name__ == "__main__":
    test_connection()