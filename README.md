# Insurance Policy RAG

A structure-aware RAG pipeline for parsing and retrieving information
from complex health insurance policy documents.

## Current Pipeline

PDF
→ PyMuPDF heading detection
→ Docling structured parsing
→ Section reconstruction
→ Chapter-aware chunking
→ Embeddings
→ Vector retrieval
→ LLM response

## Parsing Strategy

The pipeline uses a hybrid parsing approach.

PyMuPDF identifies the document's major 14-point section headings and
chapter boundaries.

Docling parses the content inside those boundaries while preserving:

- Paragraphs
- Subheadings
- Lists
- Tables
- Reading order

Each major section is treated as an atomic unit.

## Chunking Strategy

Chunks never cross chapter boundaries.

A chunk may contain one or more complete sections, but a section is
never split between chunks.

The target maximum chunk size is 750 tokens. If one section itself
exceeds 750 tokens, the complete section is retained as a single
oversized chunk.

Current output:

- 6 chapters
- 105 structured sections
- 38 final chunks

## Next Steps

- Generate embeddings
- Store chunks in a vector database
- Implement semantic retrieval
- Add LLM-based question answering
- Evaluate retrieval quality