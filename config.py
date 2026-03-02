"""
Configuration — RAG Pipeline
All tuneable constants live here. Nothing hardcoded in cl.py or app.py.
"""
import os
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ==================== SUPABASE ====================
SUPABASE_URL         = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
VECTOR_TABLE_NAME    = "documents"
IMAGE_STORAGE_BUCKET = "rag-images"   # must exist in your Supabase Storage (set to public)

# ==================== EMBEDDING ====================
EMBEDDING_MODEL      = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
EMBEDDING_DIMENSIONS = 2048   # Nemotron outputs 2048-dim vectors; HNSW uses halfvec cast

# ==================== LLM MODELS ====================
VISION_LLM_MODEL     = "qwen/qwen3-vl-235b-a22b-thinking"          # heavy — image summarisation
CLASSIFIER_LLM_MODEL = "meta-llama/llama-3.3-70b-instruct:free"       # fast  — taxonomy + query rewriting
TEXT_LLM_MODEL       = "arcee-ai/trinity-large-preview:free"        # smart — final answer generation

# ==================== OPENROUTER ====================
OPENROUTER_API_KEY   = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL  = "https://openrouter.ai/api/v1"

# ==================== COHERE ====================
COHERE_API_KEY       = os.getenv("COHERE_API_KEY")

# ==================== AUTH ====================
MASTER_ADMIN_KEY     = os.getenv("MASTER_ADMIN_KEY")   # used in auth.py with hmac.compare_digest

# ==================== UPLOAD BATCHING ====================
# FIX: chunked upload so OpenRouter free-tier rate limits aren't hit.
# Lower UPLOAD_BATCH_SIZE if you still get 429s; raise it if ingestion is too slow.
UPLOAD_BATCH_SIZE    = 10    # documents per batch
UPLOAD_BATCH_SLEEP_S = 2     # seconds to sleep between batches

# ==================== RETRIEVAL ====================
# How many conversation turns to include as memory in generate_answer()
CHAT_MEMORY_TURNS    = 3     # = last 6 messages (user + assistant pairs)

# ==================== LOGGING ====================
# Set LOG_LEVEL=DEBUG in your .env for verbose per-chunk output during ingestion.
# Valid values: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL            = os.getenv("LOG_LEVEL", "INFO")

# ==================== LEGACY / UI HINTS ====================
# No longer used as a hard whitelist — the taxonomy is built dynamically.
# These are just seed suggestions shown in the UI before any docs are ingested.
ALLOWED_CATEGORIES   = [
    "research_paper",
    "reference_chart",
    "technical_manual",
    "hr_policy",
]

# ==================== PIPELINE CONFIG MODELS ====================
# (kept for backwards compatibility with any code that imports these)

class RetrievalMode:
    BASIC_SIMILARITY = "basic_similarity"
    SCORE_THRESHOLD  = "score_threshold"
    MMR              = "mmr"


class TransformationConfig(BaseModel):
    num_queries:       int   = 5
    temperature:       float = 0.7
    prompt_template:   str   = (
        "You are a helpful assistant that generates alternative search queries.\n"
        "Given the original query, generate {num_queries} different versions.\n"
        "Each query on a new line, numbered. Only output the queries.\n\n"
        "Original Query: {query}\n\nAlternative Queries:"
    )


class HybridFusionConfig(BaseModel):
    vector_weight: float = 0.7
    bm25_weight:   float = 0.3
    use_rrf:       bool  = True
    rrf_k:         int   = 60


class AdvancedRetrievalConfig(BaseModel):
    mode:                    Literal["basic_similarity", "score_threshold", "mmr"] = "basic_similarity"
    similarity_threshold:    float = 0.5
    mmr_diversity_penalty:   float = 0.3
    top_k_per_query:         int   = 50
    final_top_k:             int   = 5
    use_reranker:            bool  = True
    reranker_model:          str   = "rerank-english-v3.0"
    rerank_on:               Literal["summary", "combined", "chunk"] = "combined"
    persist_summaries:       bool  = False
    use_persisted_summaries: bool  = True
    summary_budget_multiplier: float = 2.0
    summary_hard_cap:        int   = 200
    summary_style:           Literal["concise", "detailed"] = "concise"


class PipelineConfig(BaseModel):
    transformation:     TransformationConfig    = TransformationConfig()
    hybrid_fusion:      HybridFusionConfig      = HybridFusionConfig()
    advanced_retrieval: AdvancedRetrievalConfig  = AdvancedRetrievalConfig()
    chunk_size:         int                      = 3000
    overlap_size:       int                      = 500
    embedding_device:   Literal["cuda", "cpu"]   = "cuda"


DEFAULT_CONFIG = PipelineConfig(
    transformation=TransformationConfig(num_queries=5),
    hybrid_fusion=HybridFusionConfig(vector_weight=0.7, bm25_weight=0.3, use_rrf=True),
    advanced_retrieval=AdvancedRetrievalConfig(
        mode="basic_similarity",
        top_k_per_query=50,
        final_top_k=5,
        use_reranker=True,
    ),
)

# ==================== UI CONFIG ====================
UI_CONFIG = {
    "page_title":             "RAG Architect Pro",
    "page_icon":              "🔮",
    "layout":                 "wide",
    "initial_sidebar_state":  "expanded",
    "colors": {
        "primary":    "#10b981",
        "background": "#0f172a",
        "secondary":  "#06b6d4",
        "accent":     "#f59e0b",
    },
    "font_family":        "JetBrains Mono, monospace",
    "animation_duration": 0.5,
}
