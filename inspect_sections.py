import json
from pathlib import Path


STRUCTURED_PATH = Path("structured_sections.json")
OUTPUT_PATH = Path("hybrid_parsed_document.md")


def main():
    if not STRUCTURED_PATH.exists():
        raise FileNotFoundError(
            f"Could not find: {STRUCTURED_PATH.resolve()}"
        )

    data = json.loads(
        STRUCTURED_PATH.read_text(encoding="utf-8")
    )

    sections = data["sections"]

    parsed_document = "\n\n".join(
        section["content"].strip()
        for section in sections
        if section.get("content")
    )

    OUTPUT_PATH.write_text(
        parsed_document,
        encoding="utf-8"
    )

    print(
        f"Saved hybrid parsed document to: "
        f"{OUTPUT_PATH.resolve()}"
    )


if __name__ == "__main__":
    main()