from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PARSED_DATA_DIR = DATA_DIR / "parsed"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


# Input / output files (local testing)
POLICY_PDF_PATH = RAW_DATA_DIR / "policy.pdf"
DOCLING_OUTPUT_PATH = PARSED_DATA_DIR / "docling_output.json"

HEADINGS_PATH = PROCESSED_DATA_DIR / "headings_by_chapter.json"
STRUCTURED_SECTIONS_PATH = PROCESSED_DATA_DIR / "structured_sections.json"
POLICY_CHUNKS_PATH = PROCESSED_DATA_DIR / "policy_chunks.json"
POLICY_SUMMARIES_PATH = PROCESSED_DATA_DIR / "policy_heading_summaries.json"
CHUNK_EMBEDDINGS_PATH = PROCESSED_DATA_DIR / "policy_embeddings.json"
SUMMARY_EMBEDDINGS_PATH = PROCESSED_DATA_DIR / "policy_summary_embeddings.json"
INTERNAL_EMBEDDINGS_PATH = (PROCESSED_DATA_DIR/ "policy_internal_embeddings.json")


# Models
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIMENSION = 768
ANSWER_MODEL = "gpt-5-mini"

CROSS_ENCODER_MODEL = ("cross-encoder/ms-marco-MiniLM-L6-v2")


# Retrieval
W_NT = 0.5
W_SNT = 0.5

RRF_K = 60

CHUNK_TOP_K = 10
IE_TOP_K = 20

CROSS_ENCODER_TOP_K = 5
CROSS_ENCODER_THRESHOLD = -5.0


# Chunking
MAX_CHUNK_TOKENS = 750


# MongoDB
MONGODB_DB_NAME = "policy_rag"

CHUNKS_COLLECTION = "chunks"
INTERNAL_EMBEDDINGS_COLLECTION = "internal_embeddings"


# Atlas vector indexes
NT_VECTOR_INDEX = "nt_vector_index"
SNT_VECTOR_INDEX = "snt_vector_index"
IE_VECTOR_INDEX = "ie_vector_index"


# Atlas vector fields
NT_VECTOR_FIELD = "nt_embedding"
SNT_VECTOR_FIELD = "snt_embedding"
IE_VECTOR_FIELD = "ie_embedding"

VECTOR_NUM_CANDIDATES = 100