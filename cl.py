"""
RAG Ingestion & Retrieval Pipeline  (cl.py)
============================================
Run modes:
  python cl.py --ingest --pdf path/to/file.pdf
  python cl.py --ingest --pdf file.pdf --force
  python cl.py --query "your question here"
  python cl.py --ingest --pdf file.pdf --export
"""

import os
import json
import logging
import argparse
import uuid
import hashlib
import time
import re
from typing import List, Optional, Tuple

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

import config

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
    categories/audience absorb extra fields older LLM versions return.
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
    # FIX: absorb extra fields older LLM responses include — prevents Pydantic crash
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
    # FIX: was missing if/elif — model variable was undefined causing NameError
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
#  ENTITY EXTRACTION  (single definition — old duplicate removed)             #
# =========================================================================== #

def extract_document_entities(elements: list) -> DocumentGraphMetadata:
    log.info("Classifier analysing document...")
    sample_text = ""
    for el in elements[:25]:
        sample_text += el.text + "\n"
        if len(sample_text) > 2000:
            break

    existing = get_existing_categories()
    taxonomy_hint = (
        "EXISTING CATEGORIES (reuse if a good match exists):\n"
        + "\n".join(f"  - {c}" for c in existing)
        + "\nIf none fit, invent a new snake_case label.\n"
    ) if existing else "No categories yet — invent an appropriate snake_case category.\n"

    llm = _build_llm(use_classifier=True)
    structured_llm = llm.with_structured_output(DocumentGraphMetadata)

    prompt = (
        "You are a document taxonomy classifier.\n\n"
        f"{taxonomy_hint}\n"
        "RULES:\n"
        "- is_allowed=True for any document with real readable content.\n"
        "- is_allowed=False ONLY for blank, spam, or unreadable files.\n"
        "- Invent or reuse a snake_case document_type.\n"
        "- Extract key entities and write a one-sentence summary.\n\n"
        f"DOCUMENT EXCERPT:\n{sample_text}"
    )
    try:
        result = structured_llm.invoke([HumanMessage(content=prompt)])
        result.document_type = _sanitize_category(result.document_type)
        log.info("Type: %s | Allowed: %s | Summary: %s",
                 result.document_type, result.is_allowed, result.brief_summary)
        return result
    except Exception as exc:
        log.warning("Classifier failed (%s). Defaulting to general_document.", exc)
        return DocumentGraphMetadata.model_construct(
            is_allowed=True,
            document_type="general_document",
            key_entities=[],
            primary_topics=[],
            brief_summary="Classification failed — stored as general document.",
            categories=None,
            audience=None,
        )


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
    data = {"text": chunk.text, "tables": [], "images": [], "types": ["text"]}
    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
        for el in chunk.metadata.orig_elements:
            el_type = type(el).__name__
            if el_type == "Table":
                data["types"].append("table")
                data["tables"].append(getattr(el.metadata, "text_as_html", el.text))
            elif el_type == "Image":
                if (hasattr(el, "metadata") and hasattr(el.metadata, "image_base64")
                        and el.metadata.image_base64):
                    data["types"].append("image")
                    data["images"].append(el.metadata.image_base64)
    data["types"] = list(set(data["types"]))
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
    prompt = f"You are creating a searchable description for document retrieval.\n\nTEXT CONTENT:\n{text}\n"
    if tables:
        prompt += "\nTABLES (HTML):\n"
        for i, tbl in enumerate(tables, 1):
            prompt += f"Table {i}:\n{tbl}\n\n"
    prompt += (
        "\nYOUR TASK:\nWrite a comprehensive searchable description covering:\n"
        "1. Key facts, numbers, and data points\n"
        "2. Main topics and concepts\n"
        "3. Questions this content could answer\n"
        "4. Visual content analysis (charts, diagrams, patterns)\n"
        "5. Alternative search terms\n\nSEARCHABLE DESCRIPTION:"
    )
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
    file_path: str,
    file_hash: str,
    graph_data: DocumentGraphMetadata,
) -> Tuple[List[Document], List[str]]:
    log.info("Processing %d chunks...", len(chunks))
    docs:  List[Document] = []
    ids:   List[str]      = []
    filename  = os.path.basename(file_path)
    NAMESPACE = uuid.NAMESPACE_DNS

    for i, chunk in enumerate(chunks, 1):
        content  = _separate_content(chunk)
        chunk_id = str(uuid.uuid5(NAMESPACE, f"{file_hash}_chunk_{i}"))

        image_urls: List[str] = []
        image_b64_fallback: List[str] = []
        for j, img_b64 in enumerate(content["images"]):
            url = _upload_image_to_storage(img_b64, chunk_id, j)
            if url:
                image_urls.append(url)
            else:
                image_b64_fallback.append(img_b64)

        has_rich = bool(content["tables"] or content["images"])
        if has_rich:
            log.info("[%d/%d] Rich chunk %s → AI summary", i, len(chunks), content["types"])
            page_content = _ai_summary(content["text"], content["tables"], content["images"])
        else:
            log.info("[%d/%d] Text-only chunk", i, len(chunks))
            page_content = content["text"]

        doc = Document(
            page_content=page_content,
            metadata={
                "source":        filename,
                "file_hash":     file_hash,
                "document_type": graph_data.document_type,
                "entities":      graph_data.key_entities,
                "topics":        graph_data.primary_topics,
                "summary":       graph_data.brief_summary,
                "chunk_index":   i,
                "original_content": {
                    "raw_text":      content["text"],
                    "tables_html":   content["tables"],
                    "image_urls":    image_urls,
                    "images_base64": image_b64_fallback,
                },
            },
        )
        docs.append(doc)
        ids.append(chunk_id)

    log.info("%d documents ready", len(docs))
    return docs, ids


def is_file_already_ingested(file_hash: str) -> bool:
    supabase = _build_supabase_client()
    try:
        result = (
            supabase.table(config.VECTOR_TABLE_NAME)
            .select("id")
            .contains("metadata", {"file_hash": file_hash})
            .limit(1)
            .execute()
        )
        return len(result.data) > 0
    except Exception as exc:
        log.warning("Could not check for existing file: %s", exc)
        return False


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
    text_chars = sum(len(el.text) for el in elements if hasattr(el, "text"))
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

    # FIX: step 4 was missing — progress bar skipped from 3 to 5
    _progress(4, f"Chunking and processing (category: {graph_data.document_type})…")
    chunks = create_chunks(elements)
    docs, ids = process_chunks(chunks, pdf_path, file_hash, graph_data)
    if export_json:
        export_to_json(docs)

    _progress(5, f"Embedding and uploading {len(docs)} chunks…")
    upload_to_supabase(docs, ids)

    log.info("Ingestion complete! category='%s'", graph_data.document_type)
    return graph_data.document_type


# =========================================================================== #
#  RETRIEVAL                                                                   #
# =========================================================================== #

def generate_sub_queries(original_query: str) -> List[str]:
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
        log.info("%d sub-queries: %s", len(res.sub_queries), res.sub_queries)
        return res.sub_queries
    except Exception as exc:
        log.warning("Rewriter failed (%s). Using original.", exc)
        return [original_query]


def retrieve_chunks(
    query: str,
    k: int = 3,
    source_file: str = None,
    category: str = None,
) -> List[Document]:
    queries_to_run = generate_sub_queries(query)
    dynamic_k = 6 if len(queries_to_run) > 1 else 3
    fetch_k   = 10 if len(queries_to_run) > 1 else 15

    embeddings = _build_embeddings()
    supabase   = _build_supabase_client()

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
        query_vector = embeddings.embed_query(sub_query)
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
            top_n=dynamic_k,
        )
        retrieved = [
            Document(
                page_content=all_candidates[r.index]["content"],
                metadata=all_candidates[r.index].get("metadata", {}),
            )
            for r in rerank_response.results
        ]
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
        recent = chat_history[-6:]
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
        source_refs.append(f"[Source {i}] {source_name} (chunk {chunk_idx})")

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
