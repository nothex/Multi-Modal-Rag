"""
Configuration settings for RAG Architect Pro
"""
from typing import Literal
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

# ==================== SUPABASE ====================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
VECTOR_TABLE_NAME = "documents"

# ==================== EMBEDDING ====================
EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
EMBEDDING_DIMENSIONS = 2048   # Nemotron embed outputs 2048-dim vectors
EMBEDDING_DEVICE = "cuda"     # Change to "cpu" if no NVIDIA GPU

# ==================== LLM MODELS ====================

# Heavy vision model — used for summarising images/charts during ingestion
VISION_LLM_MODEL = "qwen/qwen3-vl-235b-a22b-thinking"

# Fast, cheap model — used for the document bouncer and query rewriting
# (these tasks don't need a big model; llama-3-8b is more than enough)
CLASSIFIER_LLM_MODEL = "meta-llama/llama-3-8b-instruct:free"

# Smart text model — used for final answer generation
TEXT_LLM_MODEL = "arcee-ai/trinity-large-preview:free"

# ==================== OPENROUTER ====================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ==================== COHERE ====================
COHERE_API_KEY = os.getenv("COHERE_API_KEY")


# ==================== PIPELINE CONFIGS ====================

class RetrievalMode:
    BASIC_SIMILARITY = "basic_similarity"
    SCORE_THRESHOLD  = "score_threshold"
    MMR              = "mmr"


class TransformationConfig(BaseModel):
    num_queries: int = 5
    temperature: float = 0.7
    prompt_template: str = (
        "You are a helpful assistant that generates alternative search queries.\n"
        "Given the original query, generate {num_queries} different versions that "
        "explore different semantic meanings and aspects.\n"
        "Each query should be on a new line and numbered. Only output the queries, nothing else.\n\n"
        "Original Query: {query}\n\nAlternative Queries:"
    )


class HybridFusionConfig(BaseModel):
    vector_weight: float = 0.6
    bm25_weight:   float = 0.4
    use_rrf:       bool  = True
    rrf_k:         int   = 60


class AdvancedRetrievalConfig(BaseModel):
    mode: Literal["basic_similarity", "score_threshold", "mmr"] = "basic_similarity"
    similarity_threshold:   float = 0.5
    mmr_diversity_penalty:  float = 0.3
    top_k_per_query:        int   = 50
    final_top_k:            int   = 5
    use_reranker:           bool  = False
    reranker_model:         str   = "BAAI/bge-reranker-base"
    rerank_on: Literal["summary", "combined", "chunk"] = "combined"
    persist_summaries:       bool  = False
    use_persisted_summaries: bool  = True
    summary_budget_multiplier: float = 2.0
    summary_hard_cap:        int   = 200
    # FIX: removed duplicate summary_style field (was defined twice — Python
    # silently kept the last one, but it's confusing and a Pydantic warning)
    summary_style: Literal["concise", "detailed"] = "concise"


class PipelineConfig(BaseModel):
    transformation:    TransformationConfig    = TransformationConfig()
    hybrid_fusion:     HybridFusionConfig      = HybridFusionConfig()
    advanced_retrieval: AdvancedRetrievalConfig = AdvancedRetrievalConfig()
    chunk_size:        int                     = 3000
    overlap_size:      int                     = 500
    embedding_device:  Literal["cuda", "cpu"]  = "cuda"


# ==================== UI CONFIG ====================
UI_CONFIG = {
    "page_title": "RAG Architect Pro",
    "page_icon": "🔮",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "colors": {
        "primary":    "#10b981",
        "background": "#0f172a",
        "secondary":  "#06b6d4",
        "accent":     "#f59e0b",
    },
    "font_family": "JetBrains Mono, monospace",
    "animation_duration": 0.5,
}

# ==================== DOCUMENT CONFIG ====================
SUPPORTED_FORMATS = ["pdf", "docx", "txt", "xlsx", "csv", "json", "md"]
DEFAULT_DOCS_PATH  = "./docs"

# ==================== DEFAULT PIPELINE CONFIG ====================
DEFAULT_CONFIG = PipelineConfig(
    transformation=TransformationConfig(num_queries=5),
    hybrid_fusion=HybridFusionConfig(vector_weight=0.6, bm25_weight=0.4, use_rrf=True),
    advanced_retrieval=AdvancedRetrievalConfig(
        mode="basic_similarity",
        top_k_per_query=50,
        final_top_k=5,
        use_reranker=False,
    ),
)
