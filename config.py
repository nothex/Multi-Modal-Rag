"""
Configuration settings for RAG Architect Pro
"""
from typing import Literal
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
VECTOR_TABLE_NAME = "documents" # Must match the table name you created in SQL

# Embedding Configuration
EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
EMBEDDING_DEVICE = "cuda" # Change to "cpu" if you don't have an NVIDIA GPU

# OpenRouter / LLM Configuration
# The heavy vision model for Ingestion (Summarizing charts/images)
# Note: I swapped to llama vision because gemini-exp was giving you a 404 earlier!
VISION_LLM_MODEL = "qwen/qwen3-vl-235b-a22b-thinking"

# --- THE BOUNCER (Auto-Classifier) CONFIG ---
# The fast model used strictly for categorizing documents
CLASSIFIER_LLM_MODEL = "meta-llama/llama-3-8b-instruct:free"

# The only document types your system is allowed to ingest.
ALLOWED_CATEGORIES = [
    "Technical Research", 
    "Educational/Academic", 
    "Business/Legal", 
    "General Knowledge",
    "Product Manual",
    "Other"
]
# The fast, smart text model for Inferencing (Answering questions)
TEXT_LLM_MODEL = "arcee-ai/trinity-large-preview:free"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Cohere Reranker Configuration
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
class RetrievalMode:
    BASIC_SIMILARITY = "basic_similarity"
    SCORE_THRESHOLD = "score_threshold"
    MMR = "mmr"

class TransformationConfig(BaseModel):
    """Multi-query transformation settings"""
    num_queries: int = 5
    temperature: float = 0.7
    prompt_template: str = """You are a helpful assistant that generates alternative search queries.
Given the original query, generate {num_queries} different versions that explore different semantic meanings and aspects.
Each query should be on a new line and numbered. Only output the queries, nothing else.

Original Query: {query}

Alternative Queries:"""

class HybridFusionConfig(BaseModel):
    """Hybrid search fusion settings"""
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    use_rrf: bool = True
    rrf_k: int = 60  # Reciprocal Rank Fusion k parameter

class AdvancedRetrievalConfig(BaseModel):
    """Advanced retrieval mode settings"""
    mode: Literal["basic_similarity", "score_threshold", "mmr"] = "basic_similarity"
    similarity_threshold: float = 0.5  # For score_threshold mode
    mmr_diversity_penalty: float = 0.3  # For MMR mode
    top_k_per_query: int = 50  # Results per query
    final_top_k: int = 5  # Final results after RRF
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"  # Optional reranker
    # Reranker options: 'summary' = use summary only, 'combined' = summary + evidence snippet, 'chunk' = full chunk text
    rerank_on: Literal["summary", "combined", "chunk"] = "combined"
    # Persist generated summaries to disk (in the vectorstore folder) to avoid re-summarizing
    persist_summaries: bool = False
    # When loading candidates, prefer persisted summaries if available
    use_persisted_summaries: bool = True
    # Summary generation budget multiplier (how many candidates to summarize relative to final_top_k)
    summary_budget_multiplier: float = 2.0
    # Absolute maximum summaries to generate regardless of multiplier
    summary_hard_cap: int = 200
    # Summary style: 'concise' for brief, intuitive explanations or 'detailed' for comprehensive metadata-rich summaries
    summary_style: Literal["concise", "detailed"] = "concise"
    # Summary style: 'concise' for brief, intuitive explanations or 'detailed' for comprehensive metadata-rich summaries
    summary_style: Literal["concise", "detailed"] = "concise"

class PipelineConfig(BaseModel):
    """Complete pipeline configuration"""
    transformation: TransformationConfig = TransformationConfig()
    hybrid_fusion: HybridFusionConfig = HybridFusionConfig()
    advanced_retrieval: AdvancedRetrievalConfig = AdvancedRetrievalConfig()
    chunk_size: int = 3000
    overlap_size: int = 500
    embedding_device: Literal["cuda", "cpu"] = "cuda"

# ==================== UI CONFIG ====================
UI_CONFIG = {
    "page_title": "RAG Architect Pro",
    "page_icon": "🔮",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "colors": {
        "primary": "#10b981",  # Emerald
        "background": "#0f172a",  # Deep slate
        "secondary": "#06b6d4",  # Cyan
        "accent": "#f59e0b",  # Amber
    },
    "font_family": "JetBrains Mono, monospace",
    "animation_duration": 0.5,
}

# ==================== DOCUMENT CONFIG ====================
SUPPORTED_FORMATS = ["pdf", "docx", "txt", "xlsx", "csv", "json", "md"]
DEFAULT_DOCS_PATH = "./docs"

# ==================== DEFAULT PIPELINE CONFIG ====================
DEFAULT_CONFIG = PipelineConfig(
    transformation=TransformationConfig(num_queries=5),
    hybrid_fusion=HybridFusionConfig(
        vector_weight=0.6,
        bm25_weight=0.4,
        use_rrf=True,
    ),
    advanced_retrieval=AdvancedRetrievalConfig(
        mode="basic_similarity",
        top_k_per_query=50,
        final_top_k=5,
        use_reranker=False,
    ),
)
