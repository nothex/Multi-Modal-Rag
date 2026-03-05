"""
RAG Ingestion & Retrieval Pipeline  (cl.py)
============================================
Run modes:
  python cl.py --ingest --pdf path/to/file.pdf
  python cl.py --ingest --pdf file.pdf --force
  python cl.py --query "your question here"
  python cl.py --ingest --pdf file.pdf --export

Improvements in this version:
  - FIX: process_chunks now uses graph_data.document_type (not .categories[0])
  - FIX: is_file_already_ingested now hits ingested_files registry table (O(1))
  - NEW: In-memory embedding cache for repeated queries (thread-safe LRU via functools)
  - NEW: ingested_files registry insert after successful upload
  - NEW: relevance_score surfaced in metadata for UI badge display
  - NEW: Source deduplication by chunk content hash (prevents near-duplicate passages)
  - NEW: Graceful empty-query guard in generate_sub_queries
  - NEW: MMR-style post-rerank diversity filter to stop one source dominating
"""

import os
import json
import logging
import argparse
import uuid
import hashlib
import time
import re
import threading
from typing import List, Optional, Tuple
from functools import lru_cache

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from pydantic import BaseModel, Field
import cohere

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.messages import HumanMessage
from supabase.client import create_client
from dotenv import load_dotenv
from classifier import DocumentClassifier
import config
import concurrent.futures
import numpy as np
load_dotenv()

# =========================================================================== #
#  STRUCTURED LOGGING                                                          #
# =========================================================================== #
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("rag_pipeline")


# =========================================================================== #
#  PYDANTIC SCHEMAS                                                            #
# =========================================================================== #

class DocumentGraphMetadata(BaseModel):
    """
    Dynamic taxonomy classification.
    All fields have safe defaults so partial LLM responses never raise.
    """
    is_allowed: bool = Field(
        default=True,
        description=(
            "True for any real document with meaningful content. "
            "False ONLY for blank/empty files, pure spam, or completely unreadable content."
        ),
    )
    document_type: str = Field(
        default="general_document",
        description=(
            "A snake_case category label. Choose from the existing list if a good match exists. "
            "Otherwise invent a concise new label e.g. 'machine_learning_paper', 'legal_contract'."
        ),
    )
    key_entities: List[str] = Field(
        default_factory=list,
        description="Names of algorithms, people, organizations, places, or technologies mentioned.",
    )
    primary_topics: List[str] = Field(
        default_factory=list,
        description="The 2-3 broad themes of the document.",
    )
    brief_summary: str = Field(
        default="No summary available.",
        description="A one-sentence summary of what this document is about.",
    )
    # Absorb extra fields older LLM responses include — prevents Pydantic crash
    categories: Optional[List[str]] = Field(default=None, exclude=True)
    audience:   Optional[str]       = Field(default=None, exclude=True)


class QueryVariants(BaseModel):
    sub_queries: List[str] = Field(
        description="1-3 highly optimized, distinct search queries broken down from the original prompt."
    )


# =========================================================================== #
#  SHARED BUILDER HELPERS                                                      #
# =========================================================================== #

def _build_llm(needs_vision: bool = False, use_classifier: bool = False) -> ChatOpenAI:
    if use_classifier:
        model = config.CLASSIFIER_LLM_MODEL
    elif needs_vision:
        model = config.VISION_LLM_MODEL
    else:
        model = config.TEXT_LLM_MODEL

    return ChatOpenAI(
        model=model,
        openai_api_key=config.OPENROUTER_API_KEY,
        openai_api_base=config.OPENROUTER_BASE_URL,
        temperature=0.1,
    )


def _build_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENROUTER_API_KEY,
        openai_api_base=config.OPENROUTER_BASE_URL,
        check_embedding_ctx_length=False,
        model_kwargs={"encoding_format": "float"},
    )


def _build_supabase_client():
    return create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)


def get_file_fingerprint(file_path: str) -> str:
    """SHA-256 hash — collision-resistant dedup key."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# =========================================================================== #
#  IN-MEMORY EMBEDDING CACHE                                                   #
#  Prevents re-embedding the same query string twice per session.              #
#  Thread-safe because CPython GIL protects dict reads/writes and              #
#  lru_cache is wrapped with a lock below for writes.                          #
# =========================================================================== #

_embed_cache: dict = {}
_embed_lock  = threading.Lock()


def get_cached_embedding(text: str) -> list:
    """Return cached embedding if available, otherwise compute and store."""
    key = hashlib.md5(text.encode()).hexdigest()
    if key in _embed_cache:
        log.debug("Embedding cache HIT for %r", text[:60])
        return _embed_cache[key]
    embeddings = _build_embeddings()
    vector = embeddings.embed_query(text)
    with _embed_lock:
        # Limit cache to 256 entries (each ~32 KB for 2048-dim float32)
        if len(_embed_cache) >= 256:
            # Evict oldest inserted key
            oldest = next(iter(_embed_cache))
            del _embed_cache[oldest]
        _embed_cache[key] = vector
    log.debug("Embedding cache MISS — cached %r", text[:60])
    return vector


# =========================================================================== #
#  DYNAMIC TAXONOMY                                                            #
# =========================================================================== #

def get_existing_categories() -> List[str]:
    """Server-side DISTINCT via get_document_types() SQL function."""
    supabase = _build_supabase_client()
    try:
        result = supabase.rpc("get_document_types", {}).execute()
        return sorted(
            row["document_type"]
            for row in (result.data or [])
            if row.get("document_type") and row["document_type"] != "unknown"
        )
    except Exception as exc:
        log.warning("Could not fetch existing categories: %s", exc)
        return []


def _sanitize_category(raw: str) -> str:
    s = raw.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s or "general_document"


# =========================================================================== #
#  ENTITY EXTRACTION                                                           #
# =========================================================================== #

def extract_document_entities(elements: list):
    """
    Classify document using the 3-stage hierarchical ensemble classifier.
    Returns a ClassificationResult that is a drop-in replacement for
    DocumentGraphMetadata — same fields accessed in process_chunks().
    """
    log.info("Classifier analysing document...")

    # Build sample text from first 50 elements (same as before)
    sample_text = ""

    # Pass 1: grab Title elements specifically (highest signal)
    for el in elements[:100]:
        el_type = type(el).__name__
        if el_type in ("Title", "Header") and hasattr(el, "text") and el.text:
            sample_text += el.text + "\n"

    # Pass 2: grab first-page body text
    for el in elements[:50]:
        el_type = type(el).__name__
        page = getattr(getattr(el, "metadata", None), "page_number", None)
        if page and page > 2:
            continue  # skip anything past page 2
        if hasattr(el, "text") and el.text and el_type not in ("Title", "Header"):
            sample_text += el.text + "\n"
        if len(sample_text) > 3000:
            break

    clf = DocumentClassifier()
    result = clf.classify(sample_text, elements)

    log.info(
        "Classification — type: '%s' | conf: %.2f | stage: %s | new: %s",
        result.document_type,
        result.confidence,
        result.stage_used,
        result.is_new_type,
    )
    if result.runner_up:
        log.info("Runner-up: '%s' (%.2f)", result.runner_up, result.runner_up_conf)

    return result


# =========================================================================== #
#  INGESTION HELPERS                                                           #
# =========================================================================== #

def partition_document(file_path: str) -> list:
    log.info("Partitioning: %s", file_path)
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
    )
    log.info("%d elements extracted", len(elements))
    return elements


def create_chunks(elements: list) -> list:
    log.info("Chunking %d elements...", len(elements))
    chunks = chunk_by_title(
        elements,
        max_characters=8000,
        new_after_n_chars=7000,
        combine_text_under_n_chars=500,
    )
    log.info("%d chunks created", len(chunks))
    return chunks


def _separate_content(chunk) -> dict:
    data = {"text": chunk.text, "tables": [], "images": [], "types": ["text"], "page_numbers": []}
    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
        for el in chunk.metadata.orig_elements:
            el_type = type(el).__name__
            pg = getattr(getattr(el, "metadata", None), "page_number", None)
            if pg is not None:
                data["page_numbers"].append(pg)
            if el_type == "Table":
                data["types"].append("table")
                data["tables"].append(getattr(el.metadata, "text_as_html", el.text))
            elif el_type == "Image":
                if (hasattr(el, "metadata") and hasattr(el.metadata, "image_base64")
                        and el.metadata.image_base64):
                    data["types"].append("image")
                    data["images"].append(el.metadata.image_base64)
    data["types"] = list(set(data["types"]))
    # Deduplicate and sort page numbers
    data["page_numbers"] = sorted(set(data["page_numbers"]))
    return data


def _upload_image_to_storage(image_b64: str, chunk_uuid: str, img_index: int) -> Optional[str]:
    """Upload to Supabase Storage bucket 'rag-images'. Returns URL or None."""
    try:
        import base64 as b64lib
        supabase  = _build_supabase_client()
        img_bytes = b64lib.b64decode(image_b64)
        path      = f"{chunk_uuid}/img_{img_index}.jpg"
        supabase.storage.from_("rag-images").upload(
            path=path,
            file=img_bytes,
            file_options={"content-type": "image/jpeg", "upsert": "true"},
        )
        url = f"{config.SUPABASE_URL}/storage/v1/object/public/rag-images/{path}"
        log.debug("Uploaded image: %s", url)
        return url
    except Exception as exc:
        log.warning("Image upload failed (%s). Keeping base64.", exc)
        return None


def _ai_summary(text: str, tables: List[str], images: List[str], max_retries: int = 3) -> str:
    vision_instruction = (
        "SYSTEM ROLE: You are a Multimodal Indexing Specialist. Your task is to extract the "
        "Semantic Essence and Structural Data from this document chunk for a high-performance RAG system.\n\n"
        "EXTRACTION HEURISTICS:\n"
        "1. NUMERICAL ANCHORING: Identify and transcribe every unique number, unit, and date. "
        "Treat high-contrast, large-font numbers as primary 'Key Performance Indicators' (KPIs).\n"
        "2. TABULAR LOGIC: If a table is present, summarize the relationship between the rows and columns "
        "so a text-search can find specific intersections (e.g., 'X value for Y category is Z').\n"
        "3. SYMBOLIC CORRELATION: Identify symbols, icons, or diagram labels and explain what "
        "concept or metric they represent.\n"
        "4. NO NARRATIVE: Avoid conversational filler. Use bulleted lists for data density.\n\n"
        "OUTPUT SCHEMA:\n"
        "- [ENTITY_DENSITY]: Key names, products, or technical terms.\n"
        "- [QUANTITATIVE_LOGS]: All raw numbers and their associated units/meanings.\n"
        "- [STRUCTURAL_SUMMARY]: A 2-sentence explanation of what this page/image is 'about'.\n"
        "- [POTENTIAL_QUERIES]: 3 specific questions this chunk could answer perfectly.\n\n"
        "INDEXING OUTPUT:"
    )
    prompt = f"{vision_instruction}\n\nTEXT CONTENT:\n{text}\n"
    if tables:
        prompt += "\nTABLES (HTML):\n"
        for i, tbl in enumerate(tables, 1):
            prompt += f"Table {i}:\n{tbl}\n\n"

    message_content: list = [{"type": "text", "text": prompt}]
    for img_b64 in images:
        message_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        })

    for attempt in range(max_retries):
        try:
            llm = _build_llm(needs_vision=bool(images))
            response = llm.invoke([HumanMessage(content=message_content)])
            return response.content
        except Exception as exc:
            err = str(exc)
            log.warning("AI Summary attempt %d/%d: %s", attempt + 1, max_retries, err[:120])
            if "404" in err:
                log.error("Vision endpoint 404 — skipping retries.")
                break
            if attempt < max_retries - 1:
                log.info("Waiting 5s...")
                time.sleep(5)

    summary = text[:300] + ("..." if len(text) > 300 else "")
    if tables: summary += f" [Contains {len(tables)} table(s)]"
    if images: summary += f" [Contains {len(images)} image(s)]"
    return summary


def process_chunks(
    chunks: list,
    elements: list,
    file_path: str,
    file_hash: str,
    graph_data: DocumentGraphMetadata,
) -> tuple[List[Document], List[str]]:
    """Convert raw unstructured chunks → LangChain Documents with parallel AI summarisation."""
    print(f"  Processing {len(chunks)} chunks...")
    docs: List[Document] = []
    ids:  List[str]      = []
    raw_filename = os.path.basename(file_path)
    filename = _extract_pdf_title(elements, raw_filename)
    NAMESPACE = uuid.NAMESPACE_DNS

    # FIX: always use document_type (string), never categories[0]
    doc_type = graph_data.document_type or "general_document"

    processed_contents = []
    rich_tasks: dict = {}

    for i, chunk in enumerate(chunks, 1):
        content = _separate_content(chunk)
        processed_contents.append((i, content))
        if content["tables"] or content["images"]:
            rich_tasks[i] = content

    ai_summaries: dict = {}
    if rich_tasks:
        print(f"  ⚡ Launching {len(rich_tasks)} AI vision tasks in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_index = {
                executor.submit(_ai_summary, c["text"], c["tables"], c["images"]): idx
                for idx, c in rich_tasks.items()
            }
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    ai_summaries[idx] = future.result()
                except Exception as exc:
                    print(f"  ⚠ Chunk {idx} generated an exception: {exc}")
                    ai_summaries[idx] = rich_tasks[idx]["text"][:300]

    for i, content in processed_contents:
        page_content = ai_summaries[i] if i in ai_summaries else content["text"]
        label = "🖼️ AI Summary" if i in ai_summaries else "📄 Text"
        print(f"    [{i}/{len(chunks)}] {label}")

        doc = Document(
            page_content=page_content,
            metadata={
                "source":         filename,
                "file_hash":      file_hash,
                # FIX: was graph_data.categories[0] which crashed when categories=None
                "document_type":  doc_type,
                "topics":         graph_data.primary_topics,
                "summary":        graph_data.brief_summary,
                "chunk_index":    i,
                "total_chunks":   len(chunks),
                "page_numbers":   content["page_numbers"],
                "original_content": json.dumps({
                    "raw_text":      content["text"],
                    "tables_html":   content["tables"],
                    "images_base64": content["images"],
                }),
            },
        )
        docs.append(doc)

        unique_string = f"{file_hash}_chunk_{i}"
        chunk_id = str(uuid.uuid5(NAMESPACE, unique_string))
        ids.append(chunk_id)

    print(f"  → {len(docs)} documents ready")
    return docs, ids


def is_file_already_ingested(file_hash: str) -> bool:
    """
    FIX: Now hits the dedicated ingested_files registry table (O(1) indexed lookup)
    instead of doing a JSONB containment scan on the full documents table.
    Falls back to the old JSONB scan if the registry table doesn't exist yet.
    """
    supabase = _build_supabase_client()
    try:
        result = (
            supabase.table("ingested_files")
            .select("id")
            .eq("file_hash", file_hash)
            .limit(1)
            .execute()
        )
        return len(result.data) > 0
    except Exception as exc:
        log.warning("ingested_files table unavailable (%s). Falling back to JSONB scan.", exc)
        try:
            result = (
                supabase.table(config.VECTOR_TABLE_NAME)
                .select("id")
                .contains("metadata", {"file_hash": file_hash})
                .limit(1)
                .execute()
            )
            return len(result.data) > 0
        except Exception as exc2:
            log.warning("Fallback dedup check also failed: %s", exc2)
            return False


def _register_ingested_file(
    file_hash: str,
    filename: str,
    document_type: str,
    chunk_count: int,
) -> None:
    """Insert a row into ingested_files registry after successful upload."""
    supabase = _build_supabase_client()
    try:
        supabase.table("ingested_files").upsert({
            "file_hash":     file_hash,
            "filename":      filename,
            "document_type": document_type,
            "chunk_count":   chunk_count,
        }, on_conflict="file_hash").execute()
        log.info("Registered in ingested_files: %s (%s)", filename, document_type)
    except Exception as exc:
        log.warning("Could not register in ingested_files: %s", exc)
# =========================================================================== #
#  ADD THESE TWO FUNCTIONS TO cl.py                                           #
#  Place them right after _register_ingested_file()                           #
# =========================================================================== #


def _apply_category_override(file_hash: str, new_category: str) -> None:
    """
    Patch document_type in all chunks belonging to this file_hash.
    Also updates ingested_files registry and refreshes materialized view.
    Safe to call any number of times — fully idempotent.
    """
    supabase = _build_supabase_client()

    # Fetch all chunks for this file
    rows = supabase.table(config.VECTOR_TABLE_NAME) \
        .select("id, metadata") \
        .eq("metadata->>file_hash", file_hash) \
        .execute()

    for row in (rows.data or []):
        meta = row["metadata"]
        meta["document_type"] = new_category
        supabase.table(config.VECTOR_TABLE_NAME) \
            .update({"metadata": meta}) \
            .eq("id", row["id"]) \
            .execute()

    # Update ingested_files registry
    supabase.table("ingested_files") \
        .update({"document_type": new_category}) \
        .eq("file_hash", file_hash) \
        .execute()

    # Refresh materialized view so sidebar filter updates immediately
    try:
        supabase.rpc("refresh_document_types_mv", {}).execute()
    except Exception as exc:
        log.warning("Could not refresh materialized view: %s", exc)

    log.info("Category override: %s… → '%s'", file_hash[:8], new_category)


def delete_document(file_hash: str) -> None:
    """
    Fully remove a document from the corpus:
      1. Delete all chunks from documents table
      2. Delete row from ingested_files registry
      3. Refresh materialized view
    Does NOT touch category_centroids — centroids are averages and
    removing one document's contribution would require recomputing from
    scratch. Since centroids are means across many docs, one deletion
    has negligible effect. Run warmup_classifier.py to fully recompute.
    """
    supabase = _build_supabase_client()

    # Step 1 — delete all chunks
    result = supabase.table(config.VECTOR_TABLE_NAME) \
        .delete() \
        .eq("metadata->>file_hash", file_hash) \
        .execute()
    deleted_count = len(result.data or [])
    log.info("Deleted %d chunks for file_hash %s…", deleted_count, file_hash[:8])

    # Step 2 — delete from registry
    supabase.table("ingested_files") \
        .delete() \
        .eq("file_hash", file_hash) \
        .execute()
    log.info("Removed from ingested_files registry.")

    # Step 3 — refresh materialized view
    try:
        supabase.rpc("refresh_document_types_mv", {}).execute()
        log.info("Materialized view refreshed after deletion.")
    except Exception as exc:
        log.warning("Could not refresh materialized view: %s", exc)



def upload_to_supabase(documents: List[Document], ids: List[str]) -> None:
    """Batched upload — respects OpenRouter free-tier rate limits."""
    BATCH_SIZE  = config.UPLOAD_BATCH_SIZE
    BATCH_SLEEP = config.UPLOAD_BATCH_SLEEP_S

    log.info("Uploading %d docs in batches of %d...", len(documents), BATCH_SIZE)
    vector_store = SupabaseVectorStore(
        embedding=_build_embeddings(),
        client=_build_supabase_client(),
        table_name=config.VECTOR_TABLE_NAME,
        query_name="match_documents",
    )
    total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_num, start in enumerate(range(0, len(documents), BATCH_SIZE), 1):
        batch_docs = documents[start : start + BATCH_SIZE]
        batch_ids  = ids[start : start + BATCH_SIZE]
        log.info("Batch %d/%d (%d docs)...", batch_num, total_batches, len(batch_docs))
        vector_store.add_documents(batch_docs, ids=batch_ids)
        if start + BATCH_SIZE < len(documents):
            time.sleep(BATCH_SLEEP)
    log.info("Upload complete.")


def export_to_json(docs: List[Document], path: str = "chunks_export.json") -> None:
    export = [
        {"chunk_id": i + 1, "enhanced_content": doc.page_content, "metadata": doc.metadata}
        for i, doc in enumerate(docs)
    ]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(export, fh, indent=2, ensure_ascii=False, default=str)
    log.info("Exported %d chunks to %s", len(export), path)


# =========================================================================== #
#  INGESTION ENTRY POINT                                                       #
# =========================================================================== #

def run_ingestion(
    pdf_path: str,
    export_json: bool = False,
    force: bool = False,
    progress_callback=None,
) -> str:
    STEPS = 5

    def _progress(step: int, msg: str):
        log.info("[%d/%d] %s", step, STEPS, msg)
        if progress_callback:
            progress_callback(step, STEPS, msg)

    log.info("=" * 50)
    log.info("Starting ingestion: %s", pdf_path)

    _progress(1, "Computing file fingerprint…")
    file_hash = get_file_fingerprint(pdf_path)
    if not force and is_file_already_ingested(file_hash):
        log.info("SKIPPING — already ingested.")
        return "already_ingested"

    _progress(2, "Partitioning PDF (OCR + layout detection)…")
    elements = partition_document(pdf_path)
    if not elements:
        raise ValueError(
            "The PDF appears blank or unreadable. "
            "If scanned, ensure tesseract-ocr is installed."
        )
    text_chars = sum(len(el.text) for el in elements if hasattr(el, "text") and el.text)
    if text_chars < 50:
        raise ValueError(
            f"PDF contains almost no readable text ({text_chars} chars). "
            "May be corrupted or image-only without OCR layer."
        )

    _progress(3, "Classifying document and building taxonomy…")
    graph_data = extract_document_entities(elements)
    if not graph_data.is_allowed:
        raise ValueError("Document rejected: appears blank, spam, or unreadable.")
    log.info("Category: '%s'", graph_data.document_type)

    _progress(4, f"Chunking and processing (category: {graph_data.document_type})…")
    chunks = create_chunks(elements)
    docs, ids = process_chunks(chunks,elements, pdf_path, file_hash, graph_data)
    if export_json:
        export_to_json(docs)

    _progress(5, f"Embedding and uploading {len(docs)} chunks…")
    upload_to_supabase(docs, ids)

    # NEW: register in the dedicated dedup table
    _register_ingested_file(
        file_hash=file_hash,
        filename=os.path.basename(pdf_path),
        document_type=graph_data.document_type,
        chunk_count=len(docs),
    )

    log.info("Ingestion complete! category='%s'", graph_data.document_type)
    return {
        "pending_review": True,
        "document_type":  graph_data.document_type,
        "filename":       os.path.basename(pdf_path),
        "file_hash":      file_hash,
    }


# =========================================================================== #
#  RETRIEVAL                                                                   #
# =========================================================================== #

def generate_sub_queries(original_query: str) -> List[str]:
    """Rewrite user query into 1-3 targeted sub-queries for better recall."""
    if not original_query or not original_query.strip():
        return ["general document information"]

    log.info("Query rewriter: %r", original_query)
    llm = _build_llm(use_classifier=True)
    structured_llm = llm.with_structured_output(QueryVariants)
    prompt = (
        "You are an expert search query optimiser.\n"
        "Break the user's question into 1-3 distinct, targeted search queries.\n"
        "If simple, return 1 optimised version. Do NOT answer it.\n\n"
        f"USER QUESTION: {original_query}"
    )
    try:
        res = structured_llm.invoke([HumanMessage(content=prompt)])
        queries = [q.strip() for q in res.sub_queries if q.strip()]
        if not queries:
            return [original_query]
        log.info("%d sub-queries: %s", len(queries), queries)
        return queries
    except Exception as exc:
        log.warning("Rewriter failed (%s). Using original.", exc)
        return [original_query]


def _diversity_filter(
    candidates: List[dict],
    top_k: int,
    max_per_source: int = 2,
) -> List[dict]:
    """
    MMR-inspired source diversity filter.
    Prevents one source from occupying all top-k slots.
    After reranking has picked the best candidates, we apply a per-source cap
    so multiple documents always contribute if available.
    """
    source_counts: dict = {}
    filtered = []
    for c in candidates:
        src = c.get("metadata", {}).get("source", "unknown")
        if source_counts.get(src, 0) < max_per_source:
            filtered.append(c)
            source_counts[src] = source_counts.get(src, 0) + 1
        if len(filtered) >= top_k:
            break
    return filtered


def retrieve_chunks(
    query: str,
    k: int = 3,
    source_file: str = None,
    category: str = None,
) -> List[Document]:
    queries_to_run = generate_sub_queries(query)
    dynamic_k = 6 if len(queries_to_run) > 1 else 3
    fetch_k   = 15 if len(queries_to_run) > 1 else 10

    # Quality bar — 0.35 is sweet spot for Cohere rerank-english-v3.0
    RELEVANCE_THRESHOLD = 0.35

    supabase = _build_supabase_client()

    filter_dict: dict = {}
    if source_file:
        filter_dict["source"] = source_file
    if category and category != "All":
        filter_dict["document_type"] = category
        log.info("Hard filter: document_type='%s'", category)

    all_candidates: list = []
    seen_ids: set = set()

    for sub_query in queries_to_run:
        log.info("Hybrid search: %r", sub_query)
        # NEW: use embedding cache to avoid re-embedding identical sub-queries
        query_vector = get_cached_embedding(sub_query)
        rpc_params = {
            "query_text":      sub_query,
            "query_embedding": query_vector,
            "match_count":     fetch_k,
            "filter":          filter_dict,
        }
        try:
            response = supabase.rpc("hybrid_search", rpc_params).execute()
        except Exception as exc:
            log.error("RPC error for %r: %s", sub_query, exc)
            continue
        for chunk in (response.data or []):
            chunk_id = chunk.get("id")
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                all_candidates.append(chunk)

    if not all_candidates:
        log.warning("No chunks found.")
        return []

    log.info("%d candidates — Cohere reranking...", len(all_candidates))
    co = cohere.Client(config.COHERE_API_KEY)
    try:
        rerank_response = co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=[c["content"] for c in all_candidates],
            top_n=min(dynamic_k * 3, len(all_candidates)),  # rerank more, filter later
        )

        # Sort by relevance descending
        ranked = sorted(
            rerank_response.results,
            key=lambda r: r.relevance_score,
            reverse=True,
        )

        # Apply relevance threshold
        above_threshold = [r for r in ranked if r.relevance_score >= RELEVANCE_THRESHOLD]
        if not above_threshold:
            log.info("No chunks above threshold %.2f; using top-%d raw.", RELEVANCE_THRESHOLD, dynamic_k)
            above_threshold = ranked[:dynamic_k]

        # Tag relevance score and collect with metadata
        scored_candidates = []
        for r in above_threshold:
            doc_data = all_candidates[r.index]
            meta = doc_data.get("metadata", {})
            meta["relevance_score"] = round(r.relevance_score, 4)
            scored_candidates.append({**doc_data, "metadata": meta})

        # NEW: apply source diversity cap (max 2 chunks per PDF)
        diverse_candidates = _diversity_filter(scored_candidates, top_k=dynamic_k, max_per_source=2)

        retrieved = [
            Document(
                page_content=c["content"],
                metadata=c.get("metadata", {}),
            )
            for c in diverse_candidates
        ]
        log.info("Dropped %d low-relevance/duplicate chunks.", len(all_candidates) - len(retrieved))

    except Exception as exc:
        log.warning("Cohere failed (%s). Raw order.", exc)
        retrieved = [
            Document(page_content=c["content"], metadata=c.get("metadata", {}))
            for c in all_candidates[:dynamic_k]
        ]

    log.info("Final %d chunks.", len(retrieved))
    return retrieved


# =========================================================================== #
#  GENERATION                                                                  #
# =========================================================================== #

def generate_answer(
    chunks: List[Document],
    query: str,
    chat_history: Optional[List[dict]] = None,
) -> str:
    if not chunks:
        return "No relevant documents were found for your query."

    memory_block = ""
    if chat_history:
        recent = chat_history[-(config.CHAT_MEMORY_TURNS * 2):]
        memory_block = "CONVERSATION HISTORY (context only):\n"
        for msg in recent:
            role = msg.get("role", "user").upper()
            memory_block += f"{role}: {msg.get('content', '')}\n"
        memory_block += "\n"

    prompt = (
        "You are a highly intelligent, direct, and concise AI enterprise assistant.\n"
        "Answer using ONLY the provided context.\n\n"
        "RULES:\n"
        "1. Be direct and concise.\n"
        "2. Cite inline: [Source N] e.g. 'Uses 512 dims [Source 1].'\n"
        "3. Answer all parts of a multi-part question.\n"
        "4. If a detail is missing, say so only for that part.\n"
        "5. If context is irrelevant: 'I'm sorry, I don't have that information.'\n\n"
        f"{memory_block}"
        f"QUESTION: {query}\n\nCONTEXT:\n"
    )

    all_images: List[str] = []
    source_refs: List[str] = []

    for i, chunk in enumerate(chunks, 1):
        prompt += f"--- Source {i} ---\n"
        meta     = chunk.metadata
        original = meta.get("original_content")
        if isinstance(original, str):
            try:
                original = json.loads(original)
            except Exception:
                original = {}
        elif not isinstance(original, dict):
            original = {}

        raw_text = original.get("raw_text", chunk.page_content)
        if raw_text:
            prompt += f"TEXT:\n{raw_text}\n\n"
        for j, tbl in enumerate(original.get("tables_html", []), 1):
            prompt += f"TABLE {j}:\n{tbl}\n\n"

        all_images.extend(original.get("images_base64", []))
        source_name = meta.get("source", f"Document {i}")
        chunk_idx   = meta.get("chunk_index", "?")
        doc_type     = meta.get("document_type", "")
        relevance   = meta.get("relevance_score")
        pages        = meta.get("page_numbers", [])
        # Build readable label
        page_str = ""
        if pages:
            page_str = f", p.{pages[0]}" if len(pages) == 1 else f", pp.{pages[0]}–{pages[-1]}"

        type_str = f" [{doc_type}]" if doc_type and doc_type != "general_document" else ""


        # Only show relevance if it's meaningful (above 0.1)
        rel_str = ""
        if relevance is not None and relevance >= 0.1:
            rel_str = f" — relevance {relevance:.2f}"
        elif relevance is not None:
            rel_str = " — low confidence"
        source_refs.append(
    f"[Source {i}] {source_name}{type_str} (chunk {chunk_idx}{page_str}{rel_str})"
)

    prompt += "\nANSWER (use [Source N] citations):"

    message_content: list = [{"type": "text", "text": prompt}]
    for img_b64 in all_images:
        message_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        })

    llm = _build_llm(needs_vision=bool(all_images))
    try:
        response = llm.invoke([HumanMessage(content=message_content)])
        answer = response.content
        if source_refs:
            answer += "\n\n---\n**Sources:**\n" + "\n".join(source_refs)
        return answer
    except Exception as exc:
        log.error("Answer generation failed: %s", exc)
        return f"Failed to generate answer: {exc}"


def run_query(
    query: str,
    k: int = 3,
    source_file: str = None,
    category: str = None,
    chat_history: Optional[List[dict]] = None,
) -> str:
    chunks = retrieve_chunks(query, k=k, source_file=source_file, category=category)
    return generate_answer(chunks, query, chat_history=chat_history)


# =========================================================================== #
#  CLI                                                                         #
# =========================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal RAG pipeline")
    parser.add_argument("--ingest",   action="store_true")
    parser.add_argument("--pdf",      type=str)
    parser.add_argument("--force",    action="store_true")
    parser.add_argument("--query",    type=str)
    parser.add_argument("--source",   type=str, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--export",   action="store_true")
    parser.add_argument("--k",        type=int, default=3)
    args = parser.parse_args()

    if args.ingest:
        if not args.pdf:
            parser.error("--pdf is required with --ingest")
        run_ingestion(args.pdf, export_json=args.export, force=args.force)

    if args.query:
        answer = run_query(args.query, k=args.k, source_file=args.source, category=args.category)
        print(f"\n💬 Answer:\n{answer}")

    if not args.ingest and not args.query:
        demo = "What is the dimensionality used in the base Transformer model?"
        print(f"\n[Demo] {demo!r}")
        print(run_query(demo))

def _apply_category_override(file_hash: str, new_category: str) -> None:
    """
    Patch document_type in all chunks belonging to this file_hash.
    Also updates ingested_files registry and centroid store.
    """
    supabase = _build_supabase_client()
    
    # Fetch all chunks for this file
    rows = supabase.table("documents") \
        .select("id, metadata") \
        .eq("metadata->>file_hash", file_hash) \
        .execute()

    for row in (rows.data or []):
        meta = row["metadata"]
        meta["document_type"] = new_category
        supabase.table("documents") \
            .update({"metadata": meta}) \
            .eq("id", row["id"]) \
            .execute()

    # Update ingested_files registry
    supabase.table("ingested_files") \
        .update({"document_type": new_category}) \
        .eq("file_hash", file_hash) \
        .execute()

    # Refresh materialized view
    try:
        supabase.rpc("refresh_document_types_mv", {}).execute()
    except Exception:
        pass

    log.info("Category override applied: %s → '%s'", file_hash[:8], new_category)

def _extract_pdf_title(elements: list, fallback_filename: str) -> str:
    """
    Try to get the real document title from:
    1. First Title element unstructured found
    2. First non-empty text line (likely a heading)
    3. Fallback to filename without extension
    """
    # Pass 1 — look for explicit Title elements
    for el in elements[:20]:
        if type(el).__name__ == "Title" and hasattr(el, "text") and el.text:
            title = el.text.strip()
            if len(title) > 3 and len(title) < 120:  # sanity bounds
                return title

    # Pass 2 — first meaningful text line
    for el in elements[:10]:
        if hasattr(el, "text") and el.text and el.text.strip():
            line = el.text.strip()
            if len(line) > 3 and len(line) < 120:
                return line

    # Fallback — clean up filename
    name = os.path.splitext(fallback_filename)[0]
    return name.replace("_", " ").replace("-", " ").title()