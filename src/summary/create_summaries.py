
from __future__ import annotations
from dotenv import load_dotenv

load_dotenv()

import json
from pathlib import Path

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


SECTIONS_PATH = Path("data/processed/structured_sections.json")
CHUNKS_PATH = Path("data/processed/policy_chunks.json")
OUTPUT_PATH = Path("data/processed/policy_heading_summaries.json")

MODEL_NAME = "gpt-5-mini"


SYSTEM_PROMPT = """
You create retrieval-oriented summaries for a RAG system.

You will receive one document heading and the content associated with that heading.

The resulting summary will later be merged with summaries from other
headings and embedded for semantic retrieval using cosine similarity.

Your task is to create a compact semantic representation of the source —
not an exhaustive restatement of it.

LENGTH TARGET

- Aim for roughly 3-6 sentences, or about 15-25% of the source length,
  whichever is shorter. Prioritize semantic coverage over strictly meeting the length target.
- The "summary" field should read as a compact narrative overview of the
  heading's content, not an enumeration of every clause or sub-condition.
  Synthesize related rules into a single compact statement where possible,
  rather than restating each one individually.

WHAT TO PRESERVE, IN PRIORITY ORDER

When compactness forces trade-offs, preserve higher-priority information
over lower-priority information:

1. Coverage/eligibility rules and the conditions attached to them
   (who/what/when/under what circumstances).
2. Numeric limits, deadlines, monetary amounts, visit/time caps, and
   other quantitative restrictions.
3. Exceptions that materially change the outcome of a general rule.
4. Domain-specific terminology and phrases likely to match future user
   queries.
5. Minor examples, illustrative detail, or restated boilerplate — these
   may be omitted or compressed to a short phrase if needed to meet the
   length target. Items in priorities 1-3 should not be dropped to save
   space.

FIDELITY RULES

- Preserve relationships between concepts rather than producing only
  keywords.
- Preserve negation and qualifying language such as "not covered",
  "only when", "unless", "may", "requires", "subject to", and
  "limited to".
- Prefer specific terminology from the source over vague wording.
- Do not add interpretations, assumptions, recommendations, or
  information that is not supported by the source.
- Do not infer that something is covered, excluded, required, or
  permitted unless the source supports that conclusion.
- If the source is ambiguous, preserve the ambiguity rather than
  resolving it.

"""

class SummaryOutput(BaseModel):
    summary: str = Field(description="Retrieval-oriented summary of the heading content.")
    key_terms: list[str] = Field(description="Important terms and phrases useful for retrieval.")


def load_sections() -> list[dict]:
    if not SECTIONS_PATH.exists():
        raise FileNotFoundError(f"Could not find: {SECTIONS_PATH.resolve()}")

    data = json.loads(SECTIONS_PATH.read_text(encoding="utf-8"))
    return data["sections"]


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Could not find: {CHUNKS_PATH.resolve()}")

    data = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return data["chunks"]


def merge_chunk_summaries(
    headings: list[str],
    heading_results: list[dict],
) -> str:

    summaries = []
    all_key_terms = []

    normalized_headings = {
        heading.strip().lower()
        for heading in headings
    }

    for item in heading_results:
        summary = item["summary"].strip()

        if summary:
            summaries.append(summary)

        for term in item["key_terms"]:
            term_clean = term.strip()

            if not term_clean:
                continue

            # Remove terms that are exactly a heading
            if term_clean.lower() in normalized_headings:
                continue

            all_key_terms.append(term_clean)

    # Remove duplicate key terms while preserving order
    unique_key_terms = list(dict.fromkeys(all_key_terms))

    # Merge heading summaries into one chunk summary
    final_summary = " ".join(summaries)

    if unique_key_terms:
        key_terms_text = ", ".join(unique_key_terms)
        final_summary += (f"\n\nKey terms: {key_terms_text}")

    return final_summary


def save_records(records: list[dict]) -> None:
    output = {
        "total_summaries": len(records),
        "records": records,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    sections = load_sections()
    chunks = load_chunks()

    print(f"Loaded {len(sections)} sections.")
    print(f"Loaded {len(chunks)} chunks.")

    # -----------------------------------------------
    # section_id -> section
    # -----------------------------------------------

    section_lookup = {
        section["section_id"]: section
        for section in sections
    }

    # -----------------------------------------------
    # LLM
    # -----------------------------------------------

    llm = ChatOpenAI(model=MODEL_NAME,temperature=0)

    structured_llm = llm.with_structured_output(SummaryOutput)

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT),
            ("human",
                """
                Heading:
                {heading}

                Content:
                {content}
                """,
            ),
        ]
    )

    chain = prompt | structured_llm

    # -----------------------------------------------
    # Generate summaries
    # -----------------------------------------------

    records = []

    for chunk_number, chunk in enumerate(chunks, start=1):
        chunk_id = f"{chunk_number:03d}"
        heading_results = []

        print(f"\nProcessing chunk {chunk_id}")

        for section_id in chunk["section_ids"]:
            section = section_lookup[section_id]

            heading = section["heading"]
            content = section["content"]

            print(f"  Summarizing: {heading}")

            result = chain.invoke({"heading": heading, "content": content})

            heading_results.append(
                {
                    "summary": result.summary,
                    "key_terms": result.key_terms,
                }
            )

        final_summary = merge_chunk_summaries(chunk["headings"],heading_results)

        record = {
            "chunk_id": chunk_id,
            "summary": final_summary,
        }

        records.append(record)

        # Save after every successful LLM call.
        save_records(records)

    print("\nSummary generation complete.")
    print(f"Total summaries: {len(records)}")
    print(f"Saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()