from pymongo.operations import SearchIndexModel

from mongodb import get_database


VECTOR_DIMENSION = 768


def main():
    db = get_database()

    chunks_collection = db["chunks"]
    ie_collection = db["internal_embeddings"]

    # --------------------------------------------------
    # NT Vector Search Index
    # --------------------------------------------------

    nt_index = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "nt_embedding",
                    "numDimensions": VECTOR_DIMENSION,
                    "similarity": "cosine",
                }
            ]
        },
        name="nt_vector_index",
        type="vectorSearch",
    )

    # --------------------------------------------------
    # SNT Vector Search Index
    # --------------------------------------------------

    snt_index = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "snt_embedding",
                    "numDimensions": VECTOR_DIMENSION,
                    "similarity": "cosine",
                }
            ]
        },
        name="snt_vector_index",
        type="vectorSearch",
    )

    # --------------------------------------------------
    # IE Vector Search Index
    #
    # chunk_id is included as a filter field so we can
    # restrict IE search to selected parent chunks later.
    # --------------------------------------------------

    ie_index = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "ie_embedding",
                    "numDimensions": VECTOR_DIMENSION,
                    "similarity": "cosine",
                },
                {
                    "type": "filter",
                    "path": "chunk_id",
                },
            ]
        },
        name="ie_vector_index",
        type="vectorSearch",
    )

    print("Creating NT vector index...")

    nt_result = chunks_collection.create_search_index(
        model=nt_index
    )

    print(f"Created: {nt_result}")

    print("Creating SNT vector index...")

    snt_result = chunks_collection.create_search_index(
        model=snt_index
    )

    print(f"Created: {snt_result}")

    print("Creating IE vector index...")

    ie_result = ie_collection.create_search_index(
        model=ie_index
    )

    print(f"Created: {ie_result}")

    print("\nVector index creation requested.")
    print(
        "Atlas may take a short time to build the indexes."
    )


if __name__ == "__main__":
    main()