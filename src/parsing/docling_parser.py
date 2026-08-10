from pathlib import Path
import json

from docling.document_converter import DocumentConverter


PDF_PATH = Path("data/raw/aetna-2500-4000-full-plan.pdf")

OUTPUT_JSON = Path("data/parsed/aetna_docling.json")
#OUTPUT_MD = Path("aetna_docling.md")


def parse_pdf(pdf_path: Path):
    converter = DocumentConverter()

    result = converter.convert(pdf_path)

    return result.document


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH.resolve()}"
        )

    print("Parsing PDF with Docling...")

    doc = parse_pdf(PDF_PATH)

    # Native Docling document representation
    doc.save_as_json(OUTPUT_JSON)

    # Human-readable version for inspection
    #doc.save_as_markdown(OUTPUT_MD)

    print(f"Saved structured JSON: {OUTPUT_JSON.resolve()}")
    #print(f"Saved Markdown:        {OUTPUT_MD.resolve()}")


if __name__ == "__main__":
    main()