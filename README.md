# Multi-Stage RAG for Health Insurance Policy Question Answering

## Background

This project originally started as a **Hack Midwest hackathon project**, where the goal was to build a useful application around a RAG pipeline using **MongoDB Atlas**. I wanted to explore a problem in healthcare where retrieval and LLMs could make complex information easier for people to understand.

While researching potential use cases, I focused on a common problem with health insurance: understanding **what services are covered, what conditions or exclusions apply, and what a patient may actually be responsible for paying**.

During this research, I found that the information needed to understand health insurance coverage is primarily spread across three types of documents:

1. **Explanation of Benefits (EOB)** — Explains how a specific patient's healthcare claim was processed, including the amount billed, what the insurance plan covered, and what the patient may owe.
2. **Summary of Benefits and Coverage (SBC)** — Provides a short, standardized overview of a health plan's major benefits, coverage levels, cost-sharing information, and common exclusions.
3. **Detailed Policy/Coverage Document** — Defines the plan's detailed coverage rules, including covered services, exclusions, limitations, medical-necessity requirements, precertification requirements, exceptions, and other policy conditions.

## Problem Statement

The project focuses on building a system where a user can ask a natural-language question about their health insurance, potentially using information from their **EOB**, and the system retrieves the relevant provisions from the detailed policy document to explain how the policy applies to their question.

Rather than simply generating an answer, the goal is to return the **relevant policy information along with its source section or page as supporting evidence**. This helps users better understand whether a service is covered, excluded, limited, or subject to specific conditions while also allowing them to verify where the answer came from in the actual policy document.

The **SBC was excluded from the primary RAG pipeline** because it is a relatively short and summarized document that can generally be provided directly to an LLM without requiring a complex retrieval architecture. I plan to address the SBC separately in a future problem statement and integrate it into this project. The detailed policy document, on the other hand, can contain significantly more coverage rules and conditions, making accurate retrieval a more suitable problem for RAG.

## Data Source

The primary data source is the **Aetna Choice POS II High Deductible Health Plan booklet**, which contains the detailed coverage rules, exclusions, limitations, and policy conditions used by the RAG pipeline.

---

## Design Philosophy

Health insurance policies can contain interconnected coverage rules, exclusions, limitations, and conditions. Traditional RAG approaches can lose this context through fixed-size chunking or retrieve semantically similar content instead of the **exact policy provision** needed to answer a question.

The system is therefore designed around three principles:

1. **Preserve document structure before embedding** — Use the original PDF structure to create meaningful policy sections instead of arbitrary fixed-size chunks.
2. **Combine multiple retrieval signals** — Use narrative embeddings, retrieval-oriented summary embeddings, and BM25 to capture both semantic and lexical relevance.
3. **Move from broad retrieval to fine-grained relevance** — Retrieve relevant chunks first, search their individual policy sections for greater precision, and use CrossEncoder reranking to identify the passages that most directly answer the query.

---

## Terminology

Before going through the architecture, a few abbreviations used throughout the pipeline:

* **NT — Narrative Text:** The complete text content of a chunk.
* **SNT — Summary of Narrative Text:** An LLM-generated retrieval-oriented summary of the chunk.
* **IE — Internal Embeddings:** Embeddings generated for the individual headings/sections contained within each chunk.

> **Note:** `section_ids` and `heading_ids` refer to the same internal section identifiers and may be used interchangeably in parts of the implementation.

---

# Architecture

```text
                        Aetna Policy Document (PDF)
                                │
                  ┌─────────────┴─────────────┐
                  │                           │
                  ▼                           ▼
               PyMuPDF                    Docling
          Heading Detection         Structured Content
                  │                           │
                  └─────────────┬─────────────┘
                                ▼
                       Structured Sections
                                │
                                ▼
                     Section-Aware Chunking
                                │
             ┌──────────────────┼──────────────────┐
             │                  │                  │
             ▼                  ▼                  ▼
       Narrative Text      LLM Summaries     Internal Sections
             │                  │                  │
             ▼                  ▼                  ▼
        NT Embeddings      SNT Embeddings      IE Embeddings
             │                  │                  │
             └──────────┬───────┴──────────┬───────┘
                        │                  │
                        ▼                  ▼
                 MongoDB Atlas      MongoDB Atlas
                chunks collection   internal_embeddings
                        │              collection
                        │
User Query              │
    │                   │
    ▼                   │
BGE Query Embedding     │
    │                   │
    ├──────────────► Atlas NT Vector Search
    │                   +
    ├──────────────► Atlas SNT Vector Search
    │                   +
    └──────────────► BM25
                        │
                        ▼
               Reciprocal Rank Fusion
                        │
                        ▼
                   Top 10 Chunks
                        │
                        ▼
             Atlas IE Vector Search
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

The first challenge is converting the original PDF into a structured representation suitable for retrieval **without losing the logical organization of the policy**.

Initially, **PyMuPDF** worked well for detecting major policy headings using font and layout information, but reconstructing the complete document structure from it was difficult. **Docling** solved this by preserving structured content such as paragraphs, lists, tables, and page provenance, but its heading detection was not as reliable for this document. This led to a hybrid parsing approach, taking advantage of what each tool does best.

The pipeline uses **PyMuPDF for heading detection and Docling for structured content extraction**. The detected headings are aligned with Docling's content to create structured policy sections containing the relevant text, substructures, and page provenance. These sections then serve as the fine-grained units for the later chunking and retrieval stages.

---

# 2. Heading/Section-Aware Chunking

Instead of using fixed-size text splitting, the pipeline treats each **complete policy section as the atomic unit**. This prevents a policy heading from being separated from the coverage rules, exclusions, or conditions that belong to it.

The chunking process follows these rules:

1. **Target chunk size: ~750 tokens** — Complete sections are grouped together while the chunk remains within this limit.
2. **Preserve complete sections** — A policy section is never split just to satisfy the token limit.
3. **Preserve chapter boundaries** — Sections from different chapters are never combined into the same chunk.
4. **Allow oversized sections** — If a single section exceeds 750 tokens, it is kept intact as its own chunk.

Each resulting chunk preserves both the larger context and references to the individual sections contained inside it:

```json
{
  "chunk_id": "001",
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
  "start_pdf_page": 5,
  "end_pdf_page": 6,
  "token_count": 734,
  "content": "..."
}
```

This structure supports the later **multi-stage retrieval process**: the complete chunk can first be used for broad retrieval, while its `section_ids` provide a direct connection back to the smaller policy sections for more precise internal retrieval.

---

# 3. Multiple Semantic Representations

A central idea behind the retrieval architecture is that **a single embedding representation may not capture every useful retrieval signal**. Each chunk is therefore represented at three different levels:

1. **NT — Narrative Text Embedding:** Represents the complete original content of the chunk.
2. **SNT — Summary Narrative Text Embedding:** Represents a condensed, retrieval-focused version of the chunk.
3. **IE — Internal Embeddings:** Represent the individual policy sections contained inside each chunk.

Together, **NT and SNT support broad chunk retrieval**, while **IE provides a second, more fine-grained retrieval layer within the selected chunks**.

## 3.1 Narrative Text Embeddings (NT)

The **NT representation** captures the complete semantic context of a chunk. The original chunk content is embedded directly using `BAAI/bge-base-en-v1.5`, without summarization or modification.

This representation is useful when the user's question is semantically similar to the language and context present in the original policy text.

```text
Chunk → Original Narrative Text → BGE → NT Embedding
```

## 3.2 Retrieval-Oriented Summary Embeddings (SNT)

The original policy text can contain important rules surrounded by examples, explanations, cross-references, and other supporting language. To create a more concentrated retrieval representation, each **heading and its associated content** is summarized using an LLM.

Unlike a general-purpose summary, the prompt is specifically designed for **retrieval**. It instructs the LLM to preserve information that could determine whether a policy provision matches a user's query, including:

* Coverage and eligibility rules
* Conditions and requirements
* Numeric limits, deadlines, and monetary amounts
* Exceptions and exclusions
* Domain-specific terminology
* Important qualifying language such as `only`, `unless`, `requires`, and `not covered`

The LLM also extracts **retrieval-oriented key terms**. The summaries generated for all headings belonging to a chunk are then combined, and their key terms are appended to create one condensed representation of that chunk.

```text
Summary: <heading 1 retrieval summary>
Summary: <heading 2 retrieval summary>
Summary: <heading 3 retrieval summary>

Key terms: medical necessity, precertification,
benefit limits, acupuncture, ...
```

This representation is embedded using the same BGE model, producing the **Summary Narrative Text (SNT) embedding**. NT captures the original context, while SNT provides a more concentrated representation of the important retrieval signals within that context.

## 3.3 Internal Embeddings (IE)

A single chunk can contain multiple distinct policy sections:

```text
Chunk 001
├── Providing covered services
├── Abortion
└── Acupuncture
```

Retrieving this chunk identifies the correct **region of the policy**, but the user's question may depend on only one of these sections. To support more precise retrieval, every individual structured section inside the chunks receives its own embedding.

```json
{
  "internal_id": "001_ie_003",
  "chunk_id": "001",
  "ie_heading": "Acupuncture",
  "ie_content": "...",
  "embedding": [...]
}
```

These **Internal Embeddings (IE)** form the second retrieval layer. After the first stage identifies the most relevant chunks using NT, SNT, and lexical retrieval, IE allows the system to search specifically within those selected chunks and identify the individual policy sections most relevant to the query.

---

# 4. First-Layer Chunk Retrieval

The first retrieval layer combines **dense semantic retrieval and sparse lexical retrieval** to create a stronger relevance signal. Dense retrieval using **NT and SNT embeddings** captures conceptual similarity, while **BM25 sparse retrieval** captures exact terminology and keyword relevance. These complementary signals are combined to rank and select the **Top 10 most relevant chunks** for the next retrieval stage.

## 4.1 Atlas NT Vector Retrieval

When a user query arrives, it is embedded using the same **BGE (`BAAI/bge-base-en-v1.5`) model** used during ingestion. The query embedding is searched against the `nt_embedding` field through the corresponding MongoDB Atlas Vector Search index.

```text
Query Embedding → nt_vector_index → NT Ranking
```

This measures similarity between the query and the original narrative content of each chunk.

## 4.2 Atlas SNT Vector Retrieval

The same query embedding is independently searched against the `snt_embedding` field, which represents the retrieval-oriented summaries created earlier.

```text
Query Embedding → snt_vector_index → SNT Ranking
```

The NT and SNT Atlas scores are then combined to calculate the chunk's dense retrieval score:

```text
Dense Score = (0.5 × NT Score) + (0.5 × SNT Score)

W_NT  = 0.5
W_SNT = 0.5
```

The current **50/50 weighting is a baseline** and can later be tuned based on retrieval evaluation results.

## 4.3 BM25 Lexical Retrieval

Alongside dense retrieval, the pipeline performs **BM25 lexical retrieval** over the chunk headings and original content.

```text
BM25 Corpus = Chunk Headings + Original Chunk Content
```

BM25 complements semantic retrieval by giving greater importance to exact and relatively rare policy terminology such as `acupuncture`, `precertification`, `medical necessity`, or `durable medical equipment`.

The first layer therefore has two complementary rankings:

```text
Dense Ranking → Semantic / conceptual relevance
BM25 Ranking  → Lexical / terminology relevance
```

## 4.4 Reciprocal Rank Fusion (RRF)

Dense and BM25 scores are produced using different scoring mechanisms and therefore are not directly comparable. Instead of adding their raw scores, the pipeline independently ranks the chunks from each retrieval method and combines those rankings using **Reciprocal Rank Fusion (RRF)**.

```text
RRF(d) =
    1 / (RRF_K + DenseRank(d))
  + 1 / (RRF_K + BM25Rank(d))

RRF_K = 60
```

RRF rewards chunks that rank strongly across both semantic and lexical retrieval while still allowing a particularly strong result from either retrieval method to contribute.

After rank fusion, the **Top 10 chunks** are retained:

```text
NT Vector Search ─┐
                  ├─→ Dense Score ─┐
SNT Vector Search ┘                │
                                   ├─→ RRF → Top 10 Chunks
BM25 Retrieval ────────────────────┘
```

These Top 10 chunks form the output of the first retrieval layer and become the candidate search space for the fine-grained **Internal Embedding retrieval stage**.

---

# 5. Fine-Grained Atlas Internal Retrieval

The **Top 10 chunks from the first retrieval layer** narrow the search to the region of the policy most likely to contain the answer. However, because each chunk can contain multiple policy sections, a second retrieval layer is used to identify the **specific sections most relevant to the query**.

The query embedding is searched against the `internal_embeddings` collection using **MongoDB Atlas Vector Search**. The search is filtered by the `chunk_id` values selected in the first layer, ensuring that only internal sections belonging to those Top 10 chunks are considered.

```text
Top 10 Chunks
      ↓
Selected Chunk IDs
      ↓
Atlas IE Vector Search
      ↓
Vector Similarity + Chunk ID Filtering
      ↓
Top 20 Internal Sections

IE_TOP_K = 20
```

This reduces the broader chunk-level results into a smaller set of **fine-grained policy passages** that can be passed to the more expensive reranking stage.

**BM25 is currently not used at the IE layer.** Initial experiments showed that IE vector retrieval was able to surface the correct fine-grained policy sections effectively, so adding lexical retrieval at this stage has been left as a future evaluation and optimization option.

---

# 6. CrossEncoder Reranking and Relevance Filtering

The previous retrieval stages use **bi-encoder embeddings**, where the query and policy passages are encoded separately. This makes retrieval efficient, but semantic similarity alone does not guarantee that a passage **directly answers the user's question**.

To improve precision, the Top IE candidates are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

For each candidate, the CrossEncoder evaluates the **query and IE passage together**:

```text
(Query, IE Passage) → CrossEncoder → Relevance Score
```

Because the query and passage are processed jointly, the CrossEncoder can better distinguish between content that is simply **related to the query** and content that **directly provides the policy information needed to answer it**.

After reranking, at most the **Top 5 passages** are retained:

```text
CROSS_ENCODER_TOP_K = 5
```

## Relevance Threshold

A final relevance threshold is applied to prevent weakly related passages from reaching the answer-generation LLM.

```text
CROSS_ENCODER_THRESHOLD = -5.0
```

If **none of the retrieved passages pass the threshold**, the pipeline does not allow the final LLM to infer an answer from weak evidence. Instead, it returns that the **available policy evidence is insufficient to determine the answer**.

This creates an additional grounding layer between retrieval and generation, reducing the chance that the LLM produces an answer when the retrieved policy does not provide sufficient supporting evidence.

---

# 7. Lost-in-the-Middle Mitigation

Even after identifying the most relevant passages, their **position inside the final LLM context can influence how effectively they are used**. In longer contexts, information placed in the middle may receive less attention than information near the beginning or end.

To reduce this effect, the pipeline rearranges the CrossEncoder-ranked passages before inserting them into the final prompt. The goal is to place the **highest-relevance passages near the boundaries of the context**.

```text
CrossEncoder Ranking:
1 → 2 → 3 → 4 → 5

Final LLM Context:
1 → 3 → 5 → 4 → 2
```

This places the **two strongest passages at opposite ends of the context**, while the remaining passages are distributed between them. The original CrossEncoder relevance scores and rankings are not changed; only the **order in which the passages are presented to the final LLM** is modified.

---

# 8. Grounded Answer Generation

The final passages that survive retrieval, CrossEncoder reranking, relevance filtering, and context ordering are passed to **`gpt-5-mini` through LangChain** for answer generation.

The final prompt is designed specifically for **grounded policy question answering**. The model is instructed to answer only from the retrieved evidence and preserve important policy distinctions such as **covered, not covered, excluded, conditionally covered, limited, required, and subject to approval**.

It must also preserve numeric limits, exceptions, requirements, time limits, qualifying language, and any ambiguity present in the original policy.

If the retrieved passages do not contain sufficient evidence, the model is instructed to **state that the available policy information is insufficient rather than filling the gap with outside knowledge**.

## End-to-End Example

For the query:

```text
Does my health plan cover acupuncture for back pain?
```

the first retrieval layer identifies the most relevant policy chunks. **Internal Embedding retrieval** then searches within those chunks and isolates the specific `Acupuncture` provision.

After CrossEncoder reranking and relevance filtering, the passage survives as supporting evidence:

```text
001_ie_003 | Acupuncture
Atlas score: 0.8615
CrossEncoder score: -1.9394
```

The surviving policy passage states that acupuncture is covered when provided by a physician as anesthesia in connection with a covered surgical procedure, while acupuncture for purposes other than anesthesia is not covered.

The grounded LLM can therefore generate an answer such as:

> **No.** The policy's *Acupuncture* section states that acupuncture is covered only when provided by a physician as a form of anesthesia in connection with a covered surgical procedure. Acupuncture for other purposes is listed as not covered.

The complete retrieval process is:

```text
User Query
    ↓
NT + SNT Dense Retrieval + BM25
    ↓
RRF → Top 10 Chunks
    ↓
IE Retrieval → Top Internal Sections
    ↓
CrossEncoder + Relevance Threshold
    ↓
Lost-in-the-Middle Ordering
    ↓
Grounded LLM
    ↓
Evidence-Based Answer
```

The architecture progressively moves from **broad chunk-level retrieval to exact policy-section retrieval**, ensuring that the final LLM receives a small set of highly relevant passages rather than the entire policy document.

---

# ⚠️ WHY THIS COMPLEX ARCHITECTURE FOR JUST 38 CHUNKS?

The current policy document produces only **38 chunks**, so a much simpler retrieval approach could likely work for this dataset alone. However, the goal is not to optimize retrieval specifically for 38 chunks, but to design and evaluate an architecture that can scale to **larger policies, multiple documents, and eventually multiple insurance plans**.

More importantly, the complexity addresses **retrieval precision rather than only dataset size**. Even with a small number of chunks, each chunk can contain multiple policy provisions, and semantic similarity alone may retrieve related content without identifying the exact rule, exclusion, or condition needed to answer the question.

The layered architecture therefore serves as a controlled baseline for evaluating **which retrieval components actually improve accuracy and which can later be simplified or removed based on evaluation results**.

---

# MongoDB Atlas Storage and Vector Retrieval

The project initially used local JSON artifacts throughout development. This was intentional because intermediate JSON representations made it easy to inspect heading extraction, structured sections, chunk boundaries, summaries, identifiers, embeddings, and retrieval behavior while the architecture was evolving.

Once the retrieval pipeline stabilized, the retrieval-ready representations were migrated to **MongoDB Atlas**. The runtime RAG pipeline now uses Atlas rather than loading complete embedding matrices from local JSON files.

The `chunks` collection stores chunk-level retrieval information:

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

Fine-grained policy sections are stored in the `internal_embeddings` collection:

```json
{
  "internal_id": "001_ie_003",
  "chunk_id": "001",
  "ie_heading": "Acupuncture",
  "ie_content": "...",
  "embedding": [...]
}
```

Atlas Vector Search indexes are created for the three vector representations:

```text
NT  → nt_vector_index
SNT → snt_vector_index
IE  → ie_vector_index
```

At query time, NT and SNT vector retrieval execute directly in Atlas. The resulting chunk rankings are combined with the local BM25 ranking using **Reciprocal Rank Fusion**. The selected chunk IDs are then used as metadata constraints for the second Atlas Vector Search over the internal-section embeddings.

This creates a clear separation between document ingestion and query-time retrieval.

```text
DOCUMENT PROCESSING / INGESTION

Document
    ↓
Structured Representations
    ↓
Embeddings
    ↓
MongoDB Atlas
    ↓
Persistent Vector Search Indexes
```

```text
QUERY PIPELINE

User Query
    ↓
Query Embedding
    ↓
Atlas Vector Search + BM25
    ↓
RRF
    ↓
Atlas IE Vector Search
    ↓
CrossEncoder
    ↓
Grounded LLM
```

Local JSON artifacts remain useful as transparent development checkpoints and reproducible intermediate outputs, but they are no longer the vector retrieval layer used by the live query pipeline.

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
│   ├── chunking/
│   │   └── create_chunks.py
│   │
│   ├── database/
│   │   ├── create_vector_indexes.py
│   │   ├── migrate_to_mongodb.py
│   │   └── mongodb.py
│   │
│   ├── embeddings/
│   │   ├── chunk_embeddings.py
│   │   ├── internal_embeddings.py
│   │   └── summary_embeddings.py
│   │
│   ├── parsing/
│   │   ├── docling_parser.py
│   │   ├── hybrid_parser.py
│   │   └── pymupdf_parser.py
│   │
│   ├── retrieval/
│   │   ├── atlas_hybrid_retrieval.py
│   │   ├── bm25_retrieval.py
│   │   ├── cross_encoding.py
│   │   ├── dense_retrieval.py
│   │   └── hybrid_retrieval.py
│   │
│   ├── summary/
│   │   └── create_summaries.py
│   │
│   ├── config.py
│   └── main.py
│
├── requirements.txt
├── .env
└── README.md
```

---

# Pipeline Components

| Component                   | Purpose                                                                                                              |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `pymupdf_parser.py`         | Extract structural heading information from the PDF                                                                  |
| `docling_parser.py`         | Parse the PDF into a richer structured representation                                                                |
| `hybrid_parser.py`          | Align heading anchors with Docling content and create structured sections                                            |
| `create_chunks.py`          | Group complete policy sections into token-aware chunks                                                               |
| `create_summaries.py`       | Generate retrieval-oriented heading summaries and key terms                                                          |
| `chunk_embeddings.py`       | Generate NT embeddings                                                                                               |
| `summary_embeddings.py`     | Generate SNT embeddings                                                                                              |
| `internal_embeddings.py`    | Generate fine-grained IE embeddings                                                                                  |
| `dense_retrieval.py`        | Evaluate NT/SNT dense retrieval during development                                                                   |
| `bm25_retrieval.py`         | Evaluate lexical retrieval                                                                                           |
| `hybrid_retrieval.py`       | Original local retrieval implementation used during development and retrieval experiments                            |
| `atlas_hybrid_retrieval.py` | Run NT/SNT Atlas Vector Search, combine dense retrieval with BM25 through RRF, and perform filtered IE Vector Search |
| `cross_encoding.py`         | Rerank Atlas IE candidates using a CrossEncoder and apply relevance filtering                                        |
| `mongodb.py`                | Manage MongoDB Atlas connections and reusable vector-search operations                                               |
| `migrate_to_mongodb.py`     | Migrate processed chunk and internal-section artifacts into MongoDB Atlas                                            |
| `create_vector_indexes.py`  | Create the Atlas Vector Search indexes used by NT, SNT, and IE retrieval                                             |
| `config.py`                 | Store shared project configuration and retrieval parameters                                                          |
| `main.py`                   | Run the complete Atlas-backed retrieval, reranking, context construction, and grounded answer-generation workflow    |

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

Create a `.env` file containing the required credentials:

```text
OPENAI_API_KEY=your_openai_api_key

MONGODB_URI=your_mongodb_atlas_connection_string
MONGODB_USERNAME=your_mongodb_username
MONGODB_PASSWORD=your_mongodb_password
```

Do not commit `.env` or database credentials to version control.

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
pymongo
```

---

# Running the RAG Pipeline

Once the document representations have been generated, migrated to MongoDB Atlas, and the vector indexes have been created, run:

```bash
python -m src.main
```

The application connects to MongoDB Atlas, loads the chunk corpus required for BM25, initializes the embedding model and CrossEncoder, and loads the grounded answer-generation model.

Document embeddings are **not regenerated** when the query pipeline starts. The precomputed NT, SNT, and IE representations are retrieved through the Atlas indexes.

The application then waits for a query:

```text
Enter query (or 'exit'):
```

Example:

```text
Enter query (or 'exit'):
Does my health plan cover acupuncture for back pain?
```

The runtime pipeline executes:

```text
Query
  ↓
BGE Query Embedding
  ↓
Atlas NT Vector Search
  +
Atlas SNT Vector Search
  +
BM25
  ↓
RRF
  ↓
Top 10 Chunks
  ↓
Atlas IE Vector Search
  ↓
CrossEncoder
  ↓
Relevance Threshold
  ↓
Lost-in-the-Middle Ordering
  ↓
Grounded LLM
  ↓
Answer
```

---

# Why Multiple Retrieval Layers?

Each retrieval stage addresses a different potential failure mode:

```text
NT Embeddings
    → Preserve the meaning of the complete original chunk

SNT Embeddings
    → Provide a compressed retrieval-oriented semantic signal

BM25
    → Preserve exact terminology and lexical matches

RRF
    → Combine semantic and lexical rankings without mixing
      incompatible raw score scales

IE Embeddings
    → Move from broad chunks to individual policy provisions

CrossEncoder
    → Determine query-passage relevance jointly

Relevance Threshold
    → Prevent weak passages from reaching generation

Grounded LLM
    → Convert retrieved policy evidence into a clear answer
```

The goal is therefore not to make one retrieval mechanism perfect. Instead, several specialized stages progressively reduce the search space from the complete policy to the evidence needed for an answer.

```text
Entire Policy
      ↓
Relevant Chunks
      ↓
Relevant Sections
      ↓
Query-Relevant Passages
      ↓
Grounded Evidence
      ↓
Answer
```

---

# Evaluation and Future Work

The current architecture provides a complete working **MongoDB Atlas-backed RAG pipeline**. The next phase will focus on systematically evaluating and tuning the existing retrieval stages rather than adding additional components without evidence that they improve performance.

Planned work includes:

* Build a representative **policy question-answer evaluation dataset**
* Evaluate **Recall@K** for chunk and IE retrieval
* Tune **NT/SNT weights**, `RRF_K`, and the **CrossEncoder relevance threshold**
* Evaluate alternative embedding models and whether **BM25 at the IE layer** provides additional value
* Add retrieval, grounding, and answer-quality evaluation
* Build a complete **end-to-end ingestion pipeline** for processing and indexing new policy documents
* Extend the architecture to support **larger and multiple policy documents**
* Integrate the **Summary of Benefits and Coverage (SBC)** into the broader project as a separate component focused on benefits, cost-sharing, and plan-level cost information

The current retrieval parameters should therefore be considered **baseline values** until validated against the evaluation dataset:

```text
W_NT = 0.5
W_SNT = 0.5
RRF_K = 60
CROSS_ENCODER_THRESHOLD = -5.0
```

The longer-term goal is to combine the detailed policy retrieval provided by the current RAG architecture with **SBC-based benefit and cost information**, bringing both sides of the original health-insurance problem into a single system.

---

# Technology Stack

**Document Processing**

* PyMuPDF
* Docling

**Chunking**

* tiktoken
* section-aware custom chunking

**Embeddings**

* Sentence Transformers
* BAAI/bge-base-en-v1.5

**Sparse Retrieval**

* BM25 / `rank_bm25`

**Rank Fusion**

* Reciprocal Rank Fusion (RRF)

**Storage / Vector Retrieval**

* MongoDB Atlas
* MongoDB Atlas Vector Search
* PyMongo

**Reranking**

* `cross-encoder/ms-marco-MiniLM-L6-v2`

**LLM Orchestration**

* LangChain

**Summary & Answer Generation**

* OpenAI models
