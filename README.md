# Multi-Stage RAG for Health Insurance Policy Question Answering

A retrieval-augmented generation (RAG) system designed to answer questions from complex health insurance policy documents while preserving coverage rules, exclusions, conditions, limits, and other policy-specific details.

Rather than relying on a single vector search over arbitrarily split chunks, this project builds a multi-stage retrieval pipeline that combines document structure, multiple semantic representations, lexical retrieval, rank fusion, fine-grained internal retrieval, CrossEncoder reranking, and grounded LLM generation.

---

## The Problem

Health insurance policies are difficult documents for traditional RAG systems.

A policy may contain hundreds of pages describing covered services, exclusions, limitations, medical-necessity requirements, precertification rules, definitions, exceptions, and benefit conditions. A simple fixed-size chunking strategy can easily separate a heading from the rule it describes or split an important condition across multiple chunks.

Retrieval presents another challenge.

A semantic embedding may understand that *acupuncture*, *physical rehabilitation*, and *pain treatment* are related concepts, but a policy question often depends on finding the **exact policy provision**, not merely a semantically similar medical topic.

For example:

> Does my health plan cover acupuncture for back pain?

A useful retrieval system must find the specific **Acupuncture** provision and preserve qualifiers such as:

> acupuncture may only be covered under particular circumstances.

This project was built around that problem.

---

## Design Philosophy

The system follows three main ideas:

**Preserve document structure before embedding it.**

The original PDF structure is used to identify meaningful policy sections rather than blindly splitting text at fixed character boundaries.

**Use multiple retrieval signals instead of trusting a single embedding.**

Narrative text embeddings, retrieval-oriented summary embeddings, and BM25 provide complementary semantic and lexical signals.

**Move from broad retrieval to fine-grained relevance.**

The system first identifies relevant chunks for recall, then searches the individual policy sections inside those chunks for precision, and finally uses a CrossEncoder to determine which passages actually answer the query.

---

# Architecture

```text
                        Health Insurance PDF
                                 │
                  ┌──────────────┴──────────────┐
                  │                             │
                  ▼                             ▼
               PyMuPDF                       Docling
          Heading Detection            Structured Content
                  │                             │
                  └──────────────┬──────────────┘
                                 ▼
                        Structured Sections
                                 │
                                 ▼
                         Section-Aware Chunking
                                 │
             ┌───────────────────┼───────────────────┐
             │                   │                   │
             ▼                   ▼                   ▼
       Narrative Text       LLM Summaries       Internal Sections
         Embeddings               │                   │
             │                    ▼                   ▼
             │             Summary Embeddings    IE Embeddings
             │                    │                   │
             └──────────┬─────────┘                   │
                        ▼                             │
                  Dense Retrieval                     │
                        │                             │
User Query ───────────► BM25                          │
                        │                             │
                        ▼                             │
               Reciprocal Rank Fusion                 │
                        │                             │
                        ▼                             │
                  Top 10 Chunks ──────────────────────┘
                        │
                        ▼
                 IE Cosine Retrieval
                        │
                        ▼
                  Top IE Candidates
                        │
                        ▼
                    CrossEncoder
                        │
                        ▼
                 Relevance Threshold
                        │
                        ▼
              Lost-in-the-Middle Ordering
                        │
                        ▼
                   Grounded LLM
                        │
                        ▼
                    Final Answer
```

---

# 1. Structure-Aware PDF Parsing

The first challenge is converting the original PDF into a representation suitable for retrieval without destroying its logical organization.

The project uses a hybrid parsing strategy combining **PyMuPDF** and **Docling**.

### PyMuPDF

PyMuPDF is used to identify structural anchors such as policy headings using PDF layout and font information.

These headings establish the semantic boundaries of the document.

### Docling

Docling provides richer structured document extraction, preserving elements such as:

- narrative text
- lists
- section headers
- tables
- page provenance

The hybrid parser aligns the headings discovered from the PDF with Docling's structured representation.

The result is a collection of structured policy sections where each major heading is associated with the content belonging to it.

Conceptually:

```text
Heading
   ↓
Associated narrative text
   ↓
Lists / subheadings / tables
   ↓
Page provenance
```

This creates the fine-grained semantic units used later in the retrieval pipeline.

---

# 2. Section-Aware Chunking

Instead of applying fixed-size text splitting, the system groups **complete policy sections** into chunks.

The current target chunk size is approximately:

```text
MAX_CHUNK_TOKENS = 750
```

Sections are kept intact whenever possible.

If several complete sections fit inside the token budget, they can be grouped into one chunk. If a single section already exceeds the token limit, it is preserved rather than being arbitrarily split.

A chunk therefore contains metadata such as:

```json
{
  "chunk_id": "...",
  "chapter": "Coverage and exclusions",
  "headings": [
    "Providing covered services",
    "Abortion",
    "Acupuncture"
  ],
  "section_ids": [
    "section_001",
    "section_002",
    "section_003"
  ],
  "content": "..."
}
```

This gives the retrieval system both larger contextual units and access to the individual sections contained inside them.

---

# 3. Multiple Semantic Representations

A central idea in this project is that one embedding representation may not capture every useful retrieval signal.

Each chunk is therefore represented at multiple levels.

## Narrative Text Embeddings (NT)

The complete chunk content is embedded using:

```text
BAAI/bge-base-en-v1.5
```

These embeddings represent the full semantic content of the original chunk.

```text
Chunk
  ↓
Original narrative text
  ↓
BGE embedding
  ↓
NT embedding
```

---

## Retrieval-Oriented Summary Embeddings (SNT)

Long policy text may contain important concepts surrounded by examples, boilerplate, cross-references, and explanatory language.

To create a more concentrated semantic representation, each heading and its associated content is sent to an LLM.

The summary generation prompt is specifically designed for **retrieval**, rather than conversational summarization.

It prioritizes:

- coverage and eligibility rules
- conditions
- numeric limits
- deadlines
- monetary amounts
- exceptions
- exclusions
- domain-specific terminology
- qualifying language such as `only`, `unless`, `requires`, and `not covered`

The LLM also extracts retrieval-oriented key terms.

Heading-level summaries belonging to the same chunk are merged, and their key terms are appended to produce a single retrieval representation for the chunk.

Example:

```text
<summary for heading 1>

<summary for heading 2>

<summary for heading 3>

Key terms: medical necessity, precertification,
benefit limits, acupuncture, ...
```

This representation is embedded using the same BGE model, producing the **Summary Narrative Text embedding (SNT)**.

---

## Internal Embeddings (IE)

Chunks can contain several distinct policy headings.

For example:

```text
Chunk 001
├── Providing covered services
├── Abortion
└── Acupuncture
```

Retrieving the chunk identifies the correct neighborhood of the document, but the final answer may depend on only one of these sections.

Therefore, every individual structured section also receives its own embedding.

Example:

```json
{
  "internal_id": "001_ie_003",
  "chunk_id": "001",
  "ie_heading": "Acupuncture",
  "ie_content": "...",
  "embedding": [...]
}
```

These are referred to as **Internal Embeddings (IE)**.

They allow the second retrieval layer to search inside the chunks selected by the first retrieval layer.

---

# 4. Dense Chunk Retrieval

When a query arrives, it is embedded using the same BGE model.

Cosine similarity is calculated independently against both representations:

```text
cos(NT, query)

cos(SNT, query)
```

The current dense score is:

```text
Dense Score =
0.5 × cos(NT, query)
+
0.5 × cos(SNT, query)
```

Therefore:

```text
W_NT  = 0.5
W_SNT = 0.5
```

These weights are currently baseline values and have not yet been formally tuned.

---

# 5. BM25 Lexical Retrieval

Dense embeddings are useful for semantic similarity, but insurance policies frequently contain exact terminology that carries significant meaning.

For example:

```text
acupuncture
precertification
medical necessity
durable medical equipment
coordination of benefits
```

BM25 provides a complementary lexical retrieval signal.

The BM25 corpus currently consists of:

```text
chunk headings + original chunk content
```

This allows rare and highly specific terms to receive greater importance than common policy words such as `plan`, `health`, or `services`.

The goal is not to replace semantic retrieval, but to combine:

```text
Dense retrieval → conceptual similarity

BM25 → lexical / terminology similarity
```

---

# 6. Reciprocal Rank Fusion

Dense similarity scores and BM25 scores exist on different scales, so their raw scores are not directly added.

Instead, each retrieval method independently ranks all chunks.

The two rankings are then combined using **Reciprocal Rank Fusion (RRF)**:

```text
RRF(d) =
1 / (k + DenseRank(d))
+
1 / (k + BM25Rank(d))
```

The current value is:

```text
RRF_K = 60
```

RRF rewards chunks that rank strongly across both semantic and lexical retrieval while still allowing a particularly strong result from either retrieval system to contribute.

After fusion, the system keeps:

```text
TOP_K = 10
```

chunks.

This forms the first retrieval layer.

---

# 7. Fine-Grained Internal Retrieval

The top 10 chunks define the region of the policy that is most likely to contain the answer.

The system then gathers only the Internal Embeddings belonging to those chunks.

```text
Top 10 chunks
       │
       ▼
Filter IEs by chunk_id
       │
       ▼
cos(query, IE)
       │
       ▼
Rank internal sections
```

The current candidate limit is:

```text
IE_TOP_K = 20
```

This creates a smaller pool of fine-grained policy passages for more expensive reranking.

BM25 is currently **not used at the IE layer**. Initial experiments showed that IE cosine retrieval successfully surfaced the correct fine-grained policy section, so lexical retrieval at this stage has been intentionally left as a future evaluation option.

---

# 8. CrossEncoder Reranking

Bi-encoder embeddings encode the query and passages separately.

This is efficient for retrieval, but semantic similarity alone does not guarantee that a passage actually answers the question.

The candidate Internal Embeddings are therefore reranked using:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

For each candidate, the CrossEncoder receives:

```text
(query, IE passage)
```

as a pair.

Unlike the embedding retrieval stage, the CrossEncoder evaluates the query and passage jointly.

This allows it to distinguish between:

```text
"this passage discusses a related medical concept"
```

and:

```text
"this passage directly addresses the user's question"
```

The reranked passages are reduced to at most:

```text
CROSS_ENCODER_TOP_K = 5
```

---

# 9. Relevance Threshold

After CrossEncoder reranking, low-relevance passages are removed before reaching the answer-generation LLM.

The current experimental threshold is:

```text
CROSS_ENCODER_THRESHOLD = -5.0
```

This value is currently a **heuristic baseline**, not a calibrated production threshold.

A future evaluation set will be used to study score distributions across relevant and irrelevant passages and determine an appropriate threshold.

If no retrieved passages survive the threshold, the system does not ask the final LLM to infer an answer from weak evidence.

Instead, it reports that the retrieved evidence is insufficient.

---

# 10. Lost-in-the-Middle Mitigation

Long LLM contexts can make information positioned in the middle less influential than information near the beginning or end.

After reranking, the relevance order is preserved conceptually but passages are rearranged before being inserted into the final prompt.

For example:

```text
CrossEncoder relevance:

1, 2, 3, 4, 5
```

becomes:

```text
LLM context:

1, 3, 5, 4, 2
```

This places the two strongest passages at opposite boundaries of the context.

The CrossEncoder ranking itself is not modified; only the final context presentation order changes.

---

# 11. Grounded Answer Generation

The final evidence is passed to:

```text
gpt-5-mini
```

through LangChain.

The answer-generation prompt explicitly instructs the model to answer **only from the retrieved policy passages**.

The grounding rules require the model to preserve distinctions such as:

- covered
- not covered
- excluded
- conditionally covered
- limited
- required
- subject to approval

The model is also instructed to preserve:

- numeric limits
- exceptions
- requirements
- time limits
- qualifying language
- ambiguity present in the original policy

If the available passages do not provide enough information, the model must state that the evidence is insufficient rather than filling the gap with outside knowledge.

---

# Example

Query:

```text
Does my health plan cover acupuncture for back pain?
```

The fine-grained retrieval layer produced:

```text
001_ie_003 | Acupuncture
IE cosine: 0.7230
CrossEncoder: -1.9394
```

After thresholding, the Acupuncture provision was supplied to the final LLM.

Example answer:

```text
No. The policy says acupuncture is only covered when provided
by a physician as a form of anesthesia in connection with a
covered surgical procedure.

The policy lists acupuncture other than for anesthesia as a
not-covered service. Therefore, acupuncture for back pain,
when it is not being provided as anesthesia for a covered
surgery, is not covered under this provision.
```

This example illustrates the purpose of the multi-stage architecture.

The first retrieval layer finds the relevant region of the policy. Internal retrieval isolates the exact provision. The CrossEncoder removes weaker semantic matches, and the final LLM answers from the surviving policy evidence.

---

# Project Structure

```text
.
├── data/
│   ├── raw/
│   │   └── policy PDF
│   │
│   ├── parsed/
│   │   └── Docling output
│   │
│   └── processed/
│       ├── headings_by_chapter.json
│       ├── structured_sections.json
│       ├── policy_chunks.json
│       ├── policy_embeddings.json
│       ├── policy_heading_summaries.json
│       ├── policy_summary_embeddings.json
│       └── policy_internal_embeddings.json
│
├── src/
│   ├── parsing/
│   │   ├── pymupdf_parser.py
│   │   ├── docling_parser.py
│   │   └── hybrid_parser.py
│   │
│   ├── chunking/
│   │   └── create_chunks.py
│   │
│   ├── summary/
│   │   └── create_summaries.py
│   │
│   ├── embeddings/
│   │   ├── chunk_embeddings.py
│   │   ├── summary_embeddings.py
│   │   └── internal_embeddings.py
│   │
│   ├── retrieval/
│   │   ├── dense_retrieval.py
│   │   ├── bm25_retrieval.py
│   │   ├── hybrid_retrieval.py
│   │   ├── cross_encoding.py
│   │   └── rag_pipeline.py
│   │
│   └── database/
│       └── mongodb.py
│
├── requirements.txt
├── .env
└── README.md
```

---

# Pipeline Components

| Component | Purpose |
|---|---|
| `pymupdf_parser.py` | Extract structural heading information from the PDF |
| `docling_parser.py` | Parse the PDF into a richer structured representation |
| `hybrid_parser.py` | Align heading anchors with Docling content and create structured sections |
| `create_chunks.py` | Group complete policy sections into token-aware chunks |
| `create_summaries.py` | Generate retrieval-oriented heading summaries and key terms |
| `chunk_embeddings.py` | Generate NT embeddings |
| `summary_embeddings.py` | Generate SNT embeddings |
| `internal_embeddings.py` | Generate fine-grained IE embeddings |
| `dense_retrieval.py` | Evaluate NT/SNT dense retrieval |
| `bm25_retrieval.py` | Evaluate lexical retrieval |
| `hybrid_retrieval.py` | Combine dense + BM25 retrieval using RRF and perform IE retrieval |
| `cross_encoding.py` | Rerank IE candidates using a CrossEncoder and apply relevance filtering |
| `rag_pipeline.py` | Run retrieval, reranking, context construction, and final answer generation |
| `mongodb.py` | Reserved for the upcoming MongoDB Atlas integration |

---

# Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file and add the OpenAI API key:

```text
OPENAI_API_KEY=your_api_key
```

The project currently depends on:

```text
PyMuPDF
Docling
tiktoken
sentence-transformers
LangChain
python-dotenv
langchain-openai
rank-bm25
```

---

# Running the Current RAG Pipeline

Once the processed artifacts and embeddings have been generated, run:

```bash
python src/retrieval/rag_pipeline.py
```

The application loads the embedding model, CrossEncoder, stored embeddings, BM25 index, and answer-generation model.

It then waits for a query:

```text
Enter query (or 'exit'):
```

Example:

```text
Enter query (or 'exit'):
Does my health plan cover acupuncture for back pain?
```

The pipeline then executes:

```text
Query
  ↓
NT/SNT dense retrieval
  +
BM25
  ↓
RRF
  ↓
Top 10 chunks
  ↓
IE cosine retrieval
  ↓
CrossEncoder
  ↓
Threshold
  ↓
Lost-in-the-middle ordering
  ↓
Grounded LLM
  ↓
Answer
```

---

# Current Retrieval Configuration

The current configuration represents the initial working baseline:

```text
Embedding model:
BAAI/bge-base-en-v1.5

NT weight:
0.5

SNT weight:
0.5

RRF k:
60

Chunk retrieval K:
10

IE candidate K:
20

CrossEncoder:
cross-encoder/ms-marco-MiniLM-L6-v2

CrossEncoder top K:
5

CrossEncoder threshold:
-5.0

Answer model:
gpt-5-mini
```

These values should not yet be interpreted as optimized hyperparameters.

They establish a reproducible baseline from which systematic evaluation can begin.

---

# Why Multiple Retrieval Layers?

Each stage solves a different failure mode.

```text
NT embeddings
    → preserve the meaning of the complete original chunk

SNT embeddings
    → provide a compressed retrieval-oriented semantic signal

BM25
    → preserve exact terminology and lexical matches

RRF
    → combine semantic and lexical rankings without mixing
      incompatible raw score scales

IE embeddings
    → move from broad chunks to individual policy provisions

CrossEncoder
    → determine query-passage relevance jointly

Threshold
    → prevent weak passages from reaching generation

Grounded LLM
    → convert retrieved policy evidence into a clear answer
```

The goal is therefore not to make one retrieval mechanism perfect.

The goal is to let several specialized stages progressively reduce the search space:

```text
Entire policy
      ↓
Relevant chunks
      ↓
Relevant sections
      ↓
Query-relevant passages
      ↓
Grounded evidence
      ↓
Answer
```

---

# Current Storage

The current implementation intentionally uses local JSON artifacts throughout development.

This made intermediate representations easy to inspect while experimenting with:

- heading extraction
- structured parsing
- chunk boundaries
- identifiers
- summary generation
- embedding representations
- retrieval behavior

The JSON files currently act as transparent checkpoints between pipeline stages.

This is useful during experimentation, but they are not intended to remain the final production storage architecture.

---

# Next Step: MongoDB Atlas

The next major phase of the project is moving persistent retrieval data into **MongoDB Atlas**.

Instead of treating separate JSON embedding files as the runtime storage layer, the planned architecture will store retrieval-ready documents directly in MongoDB.

A chunk document can eventually combine information currently distributed across several local artifacts:

```json
{
  "chunk_id": "001",
  "chapter": "Coverage and exclusions",
  "headings": [
    "Providing covered services",
    "Abortion",
    "Acupuncture"
  ],
  "content": "...",
  "summary": "...",
  "nt_embedding": [...],
  "snt_embedding": [...]
}
```

Fine-grained internal sections can be stored separately:

```json
{
  "internal_id": "001_ie_003",
  "chunk_id": "001",
  "ie_heading": "Acupuncture",
  "ie_content": "...",
  "ie_embedding": [...]
}
```

MongoDB Atlas Vector Search can then replace the current in-memory NumPy scan for vector retrieval while preserving the higher-level retrieval architecture.

The goal is:

```text
Document processing
        ↓
MongoDB Atlas
        ↓
Vector + metadata retrieval
        ↓
RRF / IE retrieval
        ↓
CrossEncoder
        ↓
LLM
```

The existing `src/database/mongodb.py` is currently reserved for this integration.

---

# Planned Ingestion Pipeline

The current processing stages are implemented independently so they can be inspected and tested individually.

A future orchestration layer will combine them into a reproducible ingestion pipeline:

```text
PDF
 ↓
PyMuPDF heading extraction
 ↓
Docling parsing
 ↓
Hybrid structural alignment
 ↓
Structured sections
 ↓
Section-aware chunking
 ↓
Retrieval-oriented summaries
 ↓
NT embeddings
 ↓
SNT embeddings
 ↓
IE embeddings
 ↓
MongoDB Atlas
```

This ingestion process will only need to run when a document is added, changed, or intentionally re-indexed.

Normal user queries will use the already-created retrieval indexes rather than regenerating document representations.

---

# Evaluation and Future Work

The current architecture is a working baseline. The next phase will focus on systematic evaluation rather than adding retrieval components without evidence that they are needed.

Planned work includes:

- MongoDB Atlas persistence and Vector Search
- centralized project configuration
- end-to-end ingestion orchestration
- representative policy QA evaluation dataset
- Recall@K evaluation for chunk retrieval
- IE retrieval evaluation
- CrossEncoder threshold calibration
- NT/SNT weight tuning
- RRF parameter evaluation
- testing alternative embedding models
- evaluating whether BM25 provides additional value at the IE layer
- retrieval and answer-quality logging
- grounded-answer evaluation
- handling larger documents and multiple policy documents

In particular, the current values:

```text
W_NT = 0.5
W_SNT = 0.5
RRF_K = 60
CROSS_ENCODER_THRESHOLD = -5.0
```

should be treated as baseline values until validated against a representative evaluation set.

---

# Technology Stack

**Document Processing**

- PyMuPDF
- Docling

**Chunking**

- tiktoken
- section-aware custom chunking

**Embeddings**

- Sentence Transformers
- BAAI/bge-base-en-v1.5

**Sparse Retrieval**

- BM25 / `rank_bm25`

**Rank Fusion**

- Reciprocal Rank Fusion (RRF)

**Reranking**

- `cross-encoder/ms-marco-MiniLM-L6-v2`

**LLM Orchestration**

- LangChain

**Summary & Answer Generation**

- OpenAI models

**Planned Storage / Vector Retrieval**

- MongoDB Atlas
- Atlas Vector Search

---

# Status

The current version implements the complete local RAG workflow:

```text
✓ Hybrid PDF parsing
✓ Structure-aware section extraction
✓ Section-aware chunking
✓ Narrative text embeddings
✓ Retrieval-oriented LLM summaries
✓ Summary embeddings
✓ Internal section embeddings
✓ Dense NT/SNT retrieval
✓ BM25 lexical retrieval
✓ Reciprocal Rank Fusion
✓ Fine-grained IE retrieval
✓ CrossEncoder reranking
✓ Relevance thresholding
✓ Lost-in-the-middle mitigation
✓ Grounded LLM answer generation

Next:
→ MongoDB Atlas integration
→ End-to-end ingestion pipeline
→ Evaluation and hyperparameter tuning
```

The project has therefore moved beyond a basic vector-search RAG system into a layered retrieval architecture designed specifically around the structural and semantic challenges of policy documents.