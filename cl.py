"""
RAG Ingestion & Retrieval Pipeline  (cl.py)
============================================
Run modes:
  python cl.py --ingest --pdf path/to/file.pdf        # Ingest a new document
  python cl.py --ingest --pdf file.pdf --force         # Re-ingest even if exists
  python cl.py --query "your question here"            # Query stored data
  python cl.py --ingest --pdf file.pdf --export        # Also export chunks to JSON
"""

import os
import json
import argparse
import uuid
import hashlib
import time
import re
from typing import List, Optional

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
#  PYDANTIC SCHEMAS                                                            #
# =========================================================================== #

class DocumentGraphMetadata(BaseModel):
    """
    Dynamic taxonomy classification.
    document_type is no longer from a fixed list — it's chosen by the LLM
    from existing categories OR a brand new one it invents.
    is_allowed is always True for real documents; only junk (spam, blank
    pages, corrupted files) gets rejected.

    All fields have safe defaults so a partial or unexpected LLM response
    never raises a Pydantic validation error.
    """
    is_allowed: bool = Field(
        default=True,   # safe default: allow ingestion, don't silently reject
        description=(
            "True for any real document with meaningful content — papers, books, "
            "manuals, reports, guides, policies, articles, notes, etc. "
            "False ONLY for: blank/empty files, pure spam, corrupted content, "
            "or files with no readable text at all."
        )
    )
    document_type: str = Field(
        default="general_document",
        description=(
            "A snake_case category label for this document. "
            "Choose the closest match from the existing categories list if one fits well. "
            "If none fit, invent a new descriptive snake_case label "
            "(e.g. 'machine_learning_paper', 'legal_contract', 'financial_report', "
            "'history_book', 'cooking_guide'). Keep it short (1-3 words)."
        )
    )
    key_entities: List[str] = Field(
        default_factory=list,
        description="Specific names of algorithms, people, organizations, places, or technologies mentioned."
    )
    primary_topics: List[str] = Field(
        default_factory=list,
        description="The 2-3 broad themes of the document."
    )
    brief_summary: str = Field(
        default="No summary available.",
        description="A one-sentence summary of what this document is about."
    )

    # ------------------------------------------------------------------ #
    # Extra fields that older LLM responses may include.                  #
    # Declaring them here with defaults means Pydantic won't crash if the #
    # LLM returns them — they're just silently accepted and ignored.      #
    # ------------------------------------------------------------------ #
    categories: Optional[List[str]] = Field(default=None, exclude=True)
    audience:   Optional[str]       = Field(default=None, exclude=True)


class QueryVariants(BaseModel):
    """Schema for the query rewriter."""
    sub_queries: List[str] = Field(
        description="1 to 3 highly optimized, distinct search queries broken down from the original prompt."
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
    """MD5 hash of file bytes — used as a deduplication key."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


# =========================================================================== #
#  DYNAMIC TAXONOMY  — replaces the hard whitelist                            #
# =========================================================================== #

def get_existing_categories() -> List[str]:
    """
    Query Supabase for every distinct document_type already in the database.
    This is the live taxonomy — it grows automatically as new docs are added.
    Returns a sorted deduplicated list, e.g. ['hr_policy', 'research_paper', ...].
    """
    supabase = _build_supabase_client()
    try:
        # Pull just the metadata column; we only need document_type values
        result = (
            supabase.table(config.VECTOR_TABLE_NAME)
            .select("metadata")
            .execute()
        )
        categories = set()
        for row in result.data:
            meta = row.get("metadata") or {}
            dt = meta.get("document_type")
            if dt and dt != "unknown":
                categories.add(dt)
        return sorted(categories)
    except Exception as exc:
        print(f"  ⚠ Could not fetch existing categories: {exc}")
        return []


def _sanitize_category(raw: str) -> str:
    """
    Ensure the LLM-produced category is clean snake_case.
    e.g. 'Machine Learning Paper' → 'machine_learning_paper'
         'ML-paper!'              → 'ml_paper'
    """
    s = raw.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)   # replace any non-alphanumeric run with _
    s = s.strip("_")                      # remove leading/trailing underscores
    return s or "general_document"


# =========================================================================== #
#  ENTITY EXTRACTION  (Classifier + Dynamic Taxonomy)                         #
# =========================================================================== #

def extract_document_entities(elements: list) -> DocumentGraphMetadata:
    """
    1. Fetches the live category taxonomy from Supabase.
    2. Shows the LLM existing categories so similar docs cluster together.
    3. LLM either reuses a category or invents a new snake_case one.
    4. Only rejects truly empty/junk files.
    """
    print("  🧠 Classifier analysing document...")

    sample_text = ""
    for el in elements[:25]:
        sample_text += el.text + "\n"
        if len(sample_text) > 2000:
            break

    # Fetch live taxonomy so the LLM can match similar docs to existing clusters
    existing = get_existing_categories()
    if existing:
        taxonomy_hint = (
            "EXISTING CATEGORIES IN THE DATABASE (reuse one if it fits well):\n"
            + "\n".join(f"  - {c}" for c in existing)
            + "\n\nIf none of the above fit, invent a new snake_case category.\n"
        )
    else:
        taxonomy_hint = (
            "No categories exist yet — this is the first document. "
            "Invent an appropriate snake_case category.\n"
        )

    llm = _build_llm(use_classifier=True)
    structured_llm = llm.with_structured_output(DocumentGraphMetadata)

    prompt = f"""You are a document taxonomy classifier for a knowledge base system.
Your job is to read a document excerpt and assign it a category label.

{taxonomy_hint}
RULES:
- Set is_allowed=True for ANY document with real, readable content.
- Set is_allowed=False ONLY for blank files, pure spam, or completely unreadable content.
- Choose document_type from the existing list above if a good match exists.
- Otherwise create a new concise snake_case label (e.g. 'machine_learning_paper', 'legal_contract').
- Extract key entities (people, algorithms, orgs, technologies, places).
- Summarise the document in one sentence.

DOCUMENT EXCERPT:
{sample_text}
"""
    try:
        result = structured_llm.invoke([HumanMessage(content=prompt)])

        # Sanitize whatever the LLM produced into clean snake_case
        result.document_type = _sanitize_category(result.document_type)

        print(f"  → Type     : {result.document_type}")
        print(f"  → Allowed  : {result.is_allowed}")
        print(f"  → Summary  : {result.brief_summary}")
        print(f"  → Entities : {result.key_entities[:3]}")
        return result

    except Exception as exc:
        print(f"  ⚠ Classifier failed ({exc}). Defaulting to general_document.")
        # model_construct() bypasses Pydantic validation entirely —
        # guaranteed to never raise, even if the schema changes again.
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
    print(f"  Partitioning: {file_path}")
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
    )
    print(f"  → {len(elements)} elements extracted")
    return elements


def create_chunks(elements: list) -> list:
    print("  Chunking...")
    chunks = chunk_by_title(
        elements,
        max_characters=8000,
        new_after_n_chars=7000,
        combine_text_under_n_chars=500,
    )
    print(f"  → {len(chunks)} chunks created")
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
                if (
                    hasattr(el, "metadata")
                    and hasattr(el.metadata, "image_base64")
                    and el.metadata.image_base64
                ):
                    data["types"].append("image")
                    data["images"].append(el.metadata.image_base64)

    data["types"] = list(set(data["types"]))
    return data


def _ai_summary(text: str, tables: List[str], images: List[str], max_retries: int = 3) -> str:
    """Generate a searchable AI summary for rich (table / image) chunks."""
    prompt = f"""You are creating a searchable description for document retrieval.

TEXT CONTENT:
{text}
"""
    if tables:
        prompt += "\nTABLES (HTML):\n"
        for i, tbl in enumerate(tables, 1):
            prompt += f"Table {i}:\n{tbl}\n\n"

    prompt += """
YOUR TASK:
Write a comprehensive, searchable description covering:
1. Key facts, numbers, and data points from text and tables
2. Main topics and concepts
3. Questions this content could answer
4. Visual content analysis (charts, diagrams, patterns found in images)
5. Alternative search terms a user might use

Prioritise findability. Be thorough but avoid padding.

SEARCHABLE DESCRIPTION:"""

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
            print(f"      ⚠ AI Summary attempt {attempt + 1} failed: {err[:120]}")
            if "404" in err:
                print("      ❌ 404: Vision endpoint unavailable — skipping retries.")
                break
            if attempt < max_retries - 1:
                print("      ⏳ Waiting 5 s before retrying...")
                time.sleep(5)
            else:
                print("      ❌ All retries exhausted.")

    summary = text[:300] + ("..." if len(text) > 300 else "")
    if tables:
        summary += f" [Contains {len(tables)} table(s)]"
    if images:
        summary += f" [Contains {len(images)} image(s)]"
    return summary


def process_chunks(
    chunks: list,
    file_path: str,
    file_hash: str,
    graph_data: DocumentGraphMetadata,
) -> tuple[List[Document], List[str]]:
    """Convert unstructured chunks → LangChain Documents with deterministic UUIDs."""
    print(f"  Processing {len(chunks)} chunks...")
    docs: List[Document] = []
    ids: List[str] = []
    filename = os.path.basename(file_path)
    NAMESPACE = uuid.NAMESPACE_DNS

    for i, chunk in enumerate(chunks, 1):
        content = _separate_content(chunk)
        has_rich = bool(content["tables"] or content["images"])

        if has_rich:
            print(f"    [{i}/{len(chunks)}] Rich chunk {content['types']} → AI summary")
            page_content = _ai_summary(content["text"], content["tables"], content["images"])
        else:
            print(f"    [{i}/{len(chunks)}] Text-only chunk")
            page_content = content["text"]

        doc = Document(
            page_content=page_content,
            metadata={
                "source": filename,
                "file_hash": file_hash,
                "document_type": graph_data.document_type,
                "entities": graph_data.key_entities,
                "topics": graph_data.primary_topics,
                "summary": graph_data.brief_summary,
                "chunk_index": i,
                "original_content": json.dumps({
                    "raw_text": content["text"],
                    "tables_html": content["tables"],
                    "images_base64": content["images"],
                }),
            },
        )
        docs.append(doc)
        ids.append(str(uuid.uuid5(NAMESPACE, f"{file_hash}_chunk_{i}")))

    print(f"  → {len(docs)} documents ready")
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
        print(f"  ⚠ Could not check for existing file: {exc}")
        return False


def upload_to_supabase(documents: List[Document], ids: List[str]) -> None:
    print("  Embedding and uploading to Supabase...")
    vector_store = SupabaseVectorStore(
        embedding=_build_embeddings(),
        client=_build_supabase_client(),
        table_name=config.VECTOR_TABLE_NAME,
        query_name="match_documents",
    )
    vector_store.add_documents(documents, ids=ids)
    print("  → Upload complete")


def export_to_json(docs: List[Document], path: str = "chunks_export.json") -> None:
    export = [
        {
            "chunk_id": i + 1,
            "enhanced_content": doc.page_content,
            "original": json.loads(doc.metadata.get("original_content", "{}")),
        }
        for i, doc in enumerate(docs)
    ]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(export, fh, indent=2, ensure_ascii=False)
    print(f"  → Exported to {path}")


# =========================================================================== #
#  INGESTION ENTRY POINT                                                       #
# =========================================================================== #

def run_ingestion(pdf_path: str, export_json: bool = False, force: bool = False) -> None:
    print("\n🚀 Starting ingestion pipeline")
    print("=" * 50)

    file_hash = get_file_fingerprint(pdf_path)
    filename = os.path.basename(pdf_path)

    if not force:
        print(f"  Checking database for {filename}...")
        if is_file_already_ingested(file_hash):
            print("  ⏭️  SKIPPING: File already ingested. Use --force to re-process.")
            return

    elements = partition_document(pdf_path)

    # Dynamic taxonomy: no hard rejection based on category
    graph_data = extract_document_entities(elements)
    if not graph_data.is_allowed:
        raise ValueError(
            f"Document rejected: the file appears to be empty, spam, or unreadable. "
            f"Please upload a PDF with real content."
        )

    print(f"\n  📂 Category assigned: '{graph_data.document_type}'")

    chunks = create_chunks(elements)
    docs, ids = process_chunks(chunks, pdf_path, file_hash, graph_data)

    if export_json:
        export_to_json(docs)

    upload_to_supabase(docs, ids)
    print(f"\n✅ Ingestion complete! Document stored under category: '{graph_data.document_type}'")


# =========================================================================== #
#  RETRIEVAL                                                                   #
# =========================================================================== #

def generate_sub_queries(original_query: str) -> List[str]:
    print(f"\n  ✂️  Query Rewriter: {original_query!r}")
    llm = _build_llm(use_classifier=True)
    structured_llm = llm.with_structured_output(QueryVariants)

    prompt = f"""You are an expert search query optimiser.
Break the user's question into 1 to 3 distinct, highly targeted search queries.
If the question is simple, return just 1 optimised version. Do NOT answer the question.

USER QUESTION: {original_query}
"""
    try:
        res = structured_llm.invoke([HumanMessage(content=prompt)])
        print(f"  → {len(res.sub_queries)} sub-queries: {res.sub_queries}")
        return res.sub_queries
    except Exception as exc:
        print(f"  ⚠ Rewriter failed ({exc}). Using original query.")
        return [original_query]


def retrieve_chunks(
    query: str,
    k: int = 3,
    source_file: str = None,
    category: str = None,
) -> List[Document]:
    """
    Full advanced retrieval:
      1. Query rewriting  →  2. Multi-query hybrid search  →
      3. Deduplication    →  4. Cohere reranking
    """
    queries_to_run = generate_sub_queries(query)
    dynamic_k = 6 if len(queries_to_run) > 1 else 3
    fetch_k = 10 if len(queries_to_run) > 1 else 15

    embeddings = _build_embeddings()
    supabase = _build_supabase_client()

    filter_dict: dict = {}
    if source_file:
        filter_dict["source"] = source_file
    if category and category != "All":
        filter_dict["document_type"] = category
        print(f"  [!] Hard filter active: document_type = '{category}'")

    all_candidates: list = []
    seen_ids: set = set()

    for sub_query in queries_to_run:
        print(f"  🔍 Hybrid search: {sub_query!r}")
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
            print(f"  ❌ RPC error for {sub_query!r}: {exc}")
            continue

        for chunk in (response.data or []):
            chunk_id = chunk.get("id")
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                all_candidates.append(chunk)

    if not all_candidates:
        print("  ⚠ No chunks found across all sub-queries.")
        return []

    print(f"  → {len(all_candidates)} unique candidates — Cohere reranking...")

    co = cohere.Client(config.COHERE_API_KEY)
    docs_to_rerank = [c["content"] for c in all_candidates]

    try:
        rerank_response = co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=docs_to_rerank,
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
        print(f"  ⚠ Cohere reranking failed ({exc}). Using raw order.")
        retrieved = [
            Document(page_content=c["content"], metadata=c.get("metadata", {}))
            for c in all_candidates[:dynamic_k]
        ]

    print(f"  → Final {len(retrieved)} chunks selected.")
    return retrieved


# =========================================================================== #
#  GENERATION                                                                  #
# =========================================================================== #

def generate_answer(chunks: List[Document], query: str) -> str:
    if not chunks:
        return "No relevant documents were found for your query."

    prompt = f"""You are a highly intelligent, direct, and concise AI enterprise assistant.
Answer the user's question using ONLY the provided context below.

CRITICAL RULES:
1. Be direct and concise.
2. Do NOT reference document numbers or say "Based on Document X".
3. Answer all parts of a multi-part question.
4. If a specific detail is missing from the context, say so only for that part.
5. If the entire context is irrelevant, say: "I'm sorry, I don't have that information."

QUESTION: {query}

CONTEXT:
"""
    all_images: List[str] = []

    for i, chunk in enumerate(chunks, 1):
        prompt += f"--- Document {i} ---\n"
        raw = chunk.metadata.get("original_content")
        try:
            original = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, TypeError):
            original = {}

        raw_text = original.get("raw_text", chunk.page_content)
        if raw_text:
            prompt += f"TEXT:\n{raw_text}\n\n"

        for j, tbl in enumerate(original.get("tables_html", []), 1):
            prompt += f"TABLE {j}:\n{tbl}\n\n"

        all_images.extend(original.get("images_base64", []))

    prompt += "\nANSWER:"

    message_content: list = [{"type": "text", "text": prompt}]
    for img_b64 in all_images:
        message_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        })

    llm = _build_llm(needs_vision=bool(all_images))

    try:
        response = llm.invoke([HumanMessage(content=message_content)])
        return response.content
    except Exception as exc:
        return f"Failed to generate answer: {exc}"


def run_query(query: str, k: int = 3, source_file: str = None, category: str = None) -> str:
    chunks = retrieve_chunks(query, k=k, source_file=source_file, category=category)
    return generate_answer(chunks, query)


# =========================================================================== #
#  CLI ENTRY POINT                                                             #
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
        demo_query = "What is the dimensionality (dmodel) used in the base Transformer model?"
        print(f"\n[Demo] Running: {demo_query!r}")
        print(run_query(demo_query))