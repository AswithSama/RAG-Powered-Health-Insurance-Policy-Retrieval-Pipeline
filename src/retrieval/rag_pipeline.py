from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
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

from cross_encoding import (
    CROSS_ENCODER_MODEL,
    rerank_with_cross_encoder,
)


load_dotenv()

ANSWER_MODEL = "gpt-5-mini"


SYSTEM_PROMPT = """
You answer questions using retrieved evidence from a health insurance
policy.

You will receive a user question and a set of policy passages selected
by a retrieval and reranking pipeline.

GROUNDING RULES

- Answer only from the provided policy passages.
- Do not use outside knowledge to determine policy coverage.
- Do not invent missing policy rules.
- Do not assume something is covered or not covered unless the retrieved
  passages support that conclusion.
- Preserve important distinctions such as:
  covered, not covered, excluded, conditionally covered, limited,
  required, and subject to approval.
- Preserve important conditions, exceptions, numeric limits,
  time limits, requirements, and qualifying language.
- Pay particular attention to words such as:
  "only", "unless", "except", "may", "must", "requires",
  "not covered", and "limited to".
- If multiple passages apply, combine them carefully without creating
  contradictions.
- If the retrieved evidence is insufficient to answer the question,
  explicitly say that the available policy passages do not provide
  enough information to determine the answer.
- Do not make medical recommendations.
- Do not reinterpret ambiguous policy language as certain.
- When useful, mention the policy heading that supports the answer.

ANSWER STYLE

- Answer the user's question directly.
- Explain the relevant policy rule in clear language.
- Include important conditions or exceptions.
- Keep the response concise while preserving information necessary
  to understand the coverage decision.
"""


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            """
Question:
{query}

Retrieved policy passages:

{context}

Answer the question using only the retrieved policy passages.
""",
        ),
    ]
)


def order_for_lost_in_middle(results: list[dict]) -> list[dict]:
    """
    CrossEncoder results arrive in descending relevance.

    Place the strongest passage at the beginning,
    the second strongest at the end,
    the third strongest near the beginning,
    the fourth near the end, etc.

    Example:

    Input relevance:
    1, 2, 3, 4, 5

    Context order:
    1, 3, 5, 4, 2
    """

    front = []
    back = []

    for index, result in enumerate(results):
        if index % 2 == 0:
            front.append(result)
        else:
            back.append(result)

    return front + list(reversed(back))


def build_context(results: list[dict]) -> str:
    context_parts = []

    for index, result in enumerate(results, start=1):
        context_parts.append(
            f"[Passage {index}]\n"
            f"Chunk ID: {result['chunk_id']}\n"
            f"Heading: {result['ie_heading']}\n"
            f"Content:\n"
            f"{result['ie_content']}"
        )

    return "\n\n".join(context_parts)


def main():

    # --------------------------------------------------
    # Load stored embeddings/data
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
    # Load models once
    # --------------------------------------------------

    print(f"Loading embedding model: {MODEL_NAME}")
    embedding_model = SentenceTransformer(MODEL_NAME)

    print(f"Loading CrossEncoder: {CROSS_ENCODER_MODEL}")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    print(f"Loading answer model: {ANSWER_MODEL}")

    answer_llm = ChatOpenAI(
        model=ANSWER_MODEL,
        temperature=0,
    )

    answer_chain = ANSWER_PROMPT | answer_llm

    # --------------------------------------------------
    # Interactive RAG
    # --------------------------------------------------

    while True:
        query = input("\nEnter query (or 'exit'): ").strip()

        if query.lower() == "exit":
            break

        # --------------------------------------------------
        # 1. Hybrid retrieval
        #
        # NT + SNT + BM25
        # -> RRF
        # -> top 10 chunks
        # -> IE cosine
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

        # --------------------------------------------------
        # 2. CrossEncoder reranking + threshold
        # --------------------------------------------------

        reranked_results = rerank_with_cross_encoder(
            query=query,
            ie_results=ie_results,
            cross_encoder=cross_encoder,
        )

        # --------------------------------------------------
        # Guardrail: no sufficiently relevant evidence
        # --------------------------------------------------

        if not reranked_results:
            print("\nNo retrieved policy passages passed the relevance threshold.")
            print("The available evidence is insufficient to answer this question reliably.")
            continue

        # --------------------------------------------------
        # 3. Lost-in-the-middle mitigation
        # --------------------------------------------------

        ordered_results = order_for_lost_in_middle(reranked_results)

        # --------------------------------------------------
        # 4. Construct LLM context
        # --------------------------------------------------

        context = build_context(ordered_results)

        # --------------------------------------------------
        # 5. Final answer generation
        # --------------------------------------------------

        response = answer_chain.invoke(
            {
                "query": query,
                "context": context,
            }
        )

        # --------------------------------------------------
        # Output
        # --------------------------------------------------

        print("\n" + "=" * 80)
        print("FINAL ANSWER")
        print("=" * 80)

        print(response.content)

        # Temporary debugging information.
        print("\n" + "-" * 80)
        print("PASSAGES USED")
        print("-" * 80)

        for result in ordered_results:
            print(
                f"{result['internal_id']} | "
                f"{result['ie_heading']} | "
                f"IE cosine: {result['ie_score']:.4f} | "
                f"CrossEncoder: {result['cross_encoder_score']:.4f}"
            )


if __name__ == "__main__":
    main()