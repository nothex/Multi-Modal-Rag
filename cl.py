"""
RAG Ingestion & Retrieval Pipeline
===================================
Run modes:
  python rag_pipeline.py --ingest --pdf path/to/file.pdf   # Ingest a new document
  python rag_pipeline.py --query "your question here"       # Query existing data
  python rag_pipeline.py --export                           # Export chunks to JSON only
"""

import os
import json
import argparse
from typing import List
import uuid
import hashlib
import time
# --------------------------------------------------------------------------- #
# Document parsing
# --------------------------------------------------------------------------- #
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from pydantic import BaseModel, Field
import cohere
# --------------------------------------------------------------------------- #
# LangChain / Supabase
# --------------------------------------------------------------------------- #
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.messages import HumanMessage
from supabase.client import create_client
from dotenv import load_dotenv

import config  # your local config.py

load_dotenv()

# =========================================================================== #
#  HELPERS                                                                     #
# =========================================================================== #
class DocumentGraphMetadata(BaseModel):
    """The structured schema for our Graph-Lite entity extraction."""
    is_allowed: bool = Field(description="True if the document is a research paper, technical manual, or relevant policy. False if it is junk like an invoice or bill.")
    document_type: str = Field(description="A 1-3 word classification of the document type.")
    key_entities: List[str] = Field(description="Specific names of algorithms (e.g., 'Scaled Dot-Product'), organizations, technologies, or key people mentioned.")
    primary_topics: List[str] = Field(description="The 2-3 broad themes of the document.")
    brief_summary: str = Field(description="A one-sentence summary of what this document is about.")

class QueryVariants(BaseModel):
    """Schema for the Query Rewriter to output structured search queries."""
    sub_queries: List[str] = Field(description="1 to 3 highly optimized, distinct search queries broken down from the original prompt.")


def extract_document_entities(elements: list) -> DocumentGraphMetadata:
    """
    Reads the beginning of a document and extracts structured Graph-like metadata.
    Acts as both a Bouncer and a Knowledge Extractor.
    """
    print("  🧠 Entity Extractor analyzing document structure...")
    
    # Grab the first ~2000 characters (Title, Abstract, Introduction)
    sample_text = ""
    for el in elements[:25]: 
        sample_text += el.text + "\n"
        if len(sample_text) > 2000:
            break
            
    # We use Trinity or Llama 3 here because they are fast and great at JSON
    llm = _build_llm(needs_vision=False) 
    
    # Force the LLM to output exactly our Pydantic schema
    structured_llm = llm.with_structured_output(DocumentGraphMetadata)
    
    prompt = f"""Analyze the following text from the beginning of a document.
Extract the key entities, topics, and determine if it belongs in a highly technical AI/Engineering knowledge base.

TEXT TO ANALYZE:
{sample_text}
"""
    try:
        # This returns a clean Python object, not a messy string!
        extracted_data = structured_llm.invoke([HumanMessage(content=prompt)])
        
        print(f"  → Document Type: {extracted_data.document_type}")
        print(f"  → Entities Found: {extracted_data.key_entities[:3]}...")
        return extracted_data
        
    except Exception as e:
        print(f"  ⚠ Extraction failed ({e}). Defaulting to rejection.")
        # Fallback empty object that forces rejection
        return DocumentGraphMetadata(
            is_allowed=False, document_type="unknown", key_entities=[], primary_topics=[], brief_summary="Failed to parse."
        )   


def _build_llm(needs_vision: bool = False) -> ChatOpenAI:
    """Return the OpenRouter LLM defined in config."""
    return ChatOpenAI(
        model_name = config.VISION_LLM_MODEL if needs_vision else config.TEXT_LLM_MODEL,                    # e.g. "google/gemma-3-27b-it:free"
        openai_api_key=config.OPENROUTER_API_KEY,
        openai_api_base=config.OPENROUTER_BASE_URL,
        temperature=0.1,
    )


def _build_embeddings() -> OpenAIEmbeddings:
    """Return the embeddings model defined in config."""
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
    """Reads the actual bytes of the file and generates a unique MD5 hash."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()
# =========================================================================== #
#  INGESTION PIPELINE                                                          #
# =========================================================================== #

def partition_document(file_path: str) -> list:
    """Extract elements from a PDF using unstructured (hi-res strategy)."""
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
    """Chunk elements by title with sensible size limits."""
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
    """
    Split a chunk into its text, table HTML, and base64 image components.
    Returns a dict with keys: text, tables, images, types.
    """
    data = {"text": chunk.text, "tables": [], "images": [], "types": ["text"]}

    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
        for el in chunk.metadata.orig_elements:
            el_type = type(el).__name__

            if el_type == "Table":
                data["types"].append("table")
                data["tables"].append(
                    getattr(el.metadata, "text_as_html", el.text)
                )

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
    """
    Ask the LLM to produce a searchable description for a mixed-content chunk.
    Includes a graceful retry loop for rate limits/server errors before falling back.
    """
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

    message_content = [{"type": "text", "text": prompt}]
    for img_b64 in images:
        message_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        })

    # --- THE NEW GRACEFUL RETRY LOOP ---
    for attempt in range(max_retries):
        try:
            # Tell the builder if we actually need the vision model
            needs_vision = len(images) > 0
            llm = _build_llm(needs_vision=needs_vision)
            
            response = llm.invoke([HumanMessage(content=message_content)])
            return response.content
            
        except Exception as exc:
            error_msg = str(exc)
            print(f"      ⚠ AI Summary Attempt {attempt + 1} failed: {error_msg[:100]}...")
            
            # If it's a 404, retrying won't help because the model physically can't see images right now
            if "404" in error_msg:
                print("      ❌ 404 Error: Vision endpoint unavailable. Skipping retries.")
                break 
                
            # If it's a rate limit (429) or server error (503), wait and try again
            if attempt < max_retries - 1:
                print("      ⏳ Waiting 5 seconds before retrying...")
                time.sleep(5)
            else:
                print("      ❌ All retries failed. Proceeding to fallback.")

    # --- THE FALLBACK (If all retries fail or it's a 404) ---
    print("      → Using raw text fallback to keep pipeline moving.")
    summary = text[:300] + ("..." if len(text) > 300 else "")
    if tables:
        summary += f" [Contains {len(tables)} table(s)]"
    if images:
        summary += f" [Contains {len(images)} image(s)]"
    return summary


def process_chunks(chunks: list, file_path: str, file_hash: str, graph_data: DocumentGraphMetadata) -> tuple[List[Document], List[str]]:
    """Convert raw unstructured chunks into LangChain Documents with cryptographic IDs."""
    print(f"  Processing {len(chunks)} chunks...")
    docs = []
    ids = []
    filename = os.path.basename(file_path)
    NAMESPACE = uuid.NAMESPACE_DNS

    for i, chunk in enumerate(chunks, 1):
        content = _separate_content(chunk)
        has_rich = bool(content["tables"] or content["images"])

        if has_rich:
            print(f"    [{i}/{len(chunks)}] Rich chunk ({content['types']}) → generating AI summary")
            page_content = _ai_summary(content["text"], content["tables"], content["images"])
        else:
            print(f"    [{i}/{len(chunks)}] Text-only chunk")
            page_content = content["text"]

        doc = Document(
            page_content=page_content,
            metadata={
                "source": filename,       # Keep human-readable filename for filtering
                "file_hash": file_hash,   # Inject the cryptographic fingerprint
                "document_type": graph_data.document_type, # Injecting the Graph Nodes!
                "entities": graph_data.key_entities,     
                "topics": graph_data.primary_topics,
                "summary": graph_data.brief_summary,
                "chunk_index": i,
                "original_content": json.dumps({
                    "raw_text": content["text"],
                    "tables_html": content["tables"],
                    "images_base64": content["images"],
                })
            },
        )
        docs.append(doc)
        
        # SUPER IMPORTANT: Generate the deterministic ID using the HASH, not the filename
        unique_string = f"{file_hash}_chunk_{i}"
        chunk_id = str(uuid.uuid5(NAMESPACE, unique_string))
        ids.append(chunk_id)

    print(f"  → {len(docs)} documents ready")
    return docs, ids


def upload_to_supabase(documents: List[Document], ids: List[str]) -> SupabaseVectorStore:
    """Embed and upload documents to Supabase."""
    print("  Embedding and uploading to Supabase...")
    vector_store = SupabaseVectorStore(
        embedding=_build_embeddings(),
        client=_build_supabase_client(),
        table_name=config.VECTOR_TABLE_NAME,
        query_name="match_documents",
    )
    # add_documents with ids forces Supabase to overwrite if the ID already exists!
    vector_store.add_documents(documents, ids=ids)
    print("  → Upload complete")
    return vector_store


def export_to_json(docs: List[Document], path: str = "chunks_export.json") -> None:
    """Save processed chunks to a JSON file for inspection."""
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

def is_file_already_ingested(file_hash: str) -> bool:
    """Check Supabase to see if this exact file content already exists."""
    supabase = _build_supabase_client()
    try:
        # Search the JSON metadata specifically for the "file_hash" key
        result = supabase.table(config.VECTOR_TABLE_NAME) \
                         .select("id") \
                         .contains("metadata", {"file_hash": file_hash}) \
                         .limit(1) \
                         .execute()
        return len(result.data) > 0
    except Exception as e:
        print(f"  ⚠ Warning: Could not check database for existing files: {e}")
        return False
    
def run_ingestion(pdf_path: str, export_json: bool = False, force: bool = False) -> None:
    """Full ingestion pipeline: partition → graph extract → chunk → embed → upload."""
    print("\n🚀 Starting ingestion pipeline")
    print("=" * 50)
    
    # 1. Generate the fingerprint first
    file_hash = get_file_fingerprint(pdf_path)
    filename = os.path.basename(pdf_path)

    # 2. Gatekeeper Check
    if not force:
        print(f"  Checking database for fingerprint of {filename}...")
        if is_file_already_ingested(file_hash):
            print(f"  ⏭️ SKIPPING: A file with this exact content is already fully ingested.")
            print("  (If you want to re-process it, add a --force flag to your command).")
            return

    # 3. Partition First (We need the text so the Extractor can read it)
    elements = partition_document(pdf_path)
    
    # 4. ADVANCED GRAPH EXTRACTION & BOUNCER
    graph_data = extract_document_entities(elements)
    
    if not graph_data.is_allowed:
        print(f"\n❌ INGESTION ABORTED: Document '{filename}' was rejected by the system.")
        print(f"Identified as: {graph_data.document_type}")
        # If running in Streamlit, raise an error so the UI shows a red warning
        raise ValueError(f"Document rejected. Identified as {graph_data.document_type}. Please upload a valid knowledge document.")

    # 5. Process and Upload (Passing the graph_data in!)
    chunks = create_chunks(elements)
    docs, ids = process_chunks(chunks, pdf_path, file_hash, graph_data)

    if export_json:
        export_to_json(docs)

    upload_to_supabase(docs, ids)
    print("\n✅ Ingestion complete!")


# =========================================================================== #
#  RETRIEVAL & GENERATION                                                      #
# =========================================================================== #
def generate_sub_queries(original_query: str) -> List[str]:
    """Intercepts a complex user prompt and breaks it into optimized sub-queries."""
    print(f"\n  ✂️ Query Rewriter analyzing: {original_query!r}")
    
    # We use our fast text LLM
    llm = _build_llm(needs_vision=False)
    structured_llm = llm.with_structured_output(QueryVariants)
    
    prompt = f"""You are an expert search engine query optimizer. 
A user has asked a complex or multi-part question. Your job is to break it down into 1 to 3 distinct, highly targeted search queries.
If the question is simple and only asks one thing, just return 1 optimized version of it. Do not answer the question.

USER QUESTION: {original_query}
"""
    try:
        res = structured_llm.invoke([HumanMessage(content=prompt)])
        print(f"  → Rewrote into {len(res.sub_queries)} queries: {res.sub_queries}")
        return res.sub_queries
    except Exception as e:
        print(f"  ⚠ Rewriter failed ({e}). Using original query.")
        return [original_query]
    

def retrieve_chunks(query: str, k: int = 3, source_file: str = None, category: str = None) -> List[Document]:
    """Advanced Retrieval: Query Rewriting → Multi-Hybrid Search → Deduplication → Cohere Reranking."""
    
    # 1. REWRITE THE QUERY
    queries_to_run = generate_sub_queries(query)
    # If the rewriter split the query, we need more "slots" in the final result
    dynamic_k = 6 if len(queries_to_run) > 1 else 3
    print(f"  ⚡ Dynamic Context: Using k={dynamic_k} for {len(queries_to_run)} sub-queries.")
    embeddings = _build_embeddings()
    supabase = _build_supabase_client()
    
    # Build the filter dict
    filter_dict = {}
    if source_file:
        filter_dict["source"] = source_file
    if category and category != "All":
        filter_dict["document_type"] = category
        print(f"  [!] Applying Hard Database Filter: Only searching '{category}' documents.")

    # 2. MULTI-SEARCH & POOLING
    all_candidates = []
    seen_ids = set() # To prevent duplicate chunks
    
    # We fetch fewer chunks per query since we are running multiple queries
    fetch_k = 15 if len(queries_to_run) == 1 else 10
    
    for sub_query in queries_to_run:
        print(f"  🔍 Running Hybrid Search for sub-query: {sub_query!r}")
        query_vector = embeddings.embed_query(sub_query)
        
        rpc_params = {
            "query_text": sub_query,
            "query_embedding": query_vector,
            "match_count": fetch_k,
            "filter": filter_dict
        }
        
        response = supabase.rpc("hybrid_search", rpc_params).execute()
        
        if response.data:
            for chunk in response.data:
                # Deduplication logic (Don't add the same chunk twice!)
                if chunk["id"] not in seen_ids:
                    seen_ids.add(chunk["id"])
                    all_candidates.append(chunk)

    if not all_candidates:
        print("  → ⚠ No chunks found in the database across any sub-queries.")
        return []

    print(f"  → Pooled {len(all_candidates)} unique candidates. Sending to Cohere...")

    # 3. CLOUD RERANKING (COHERE)
    # Important: We tell Cohere to grade the chunks based on the ORIGINAL user query!
    docs_to_rerank = [item["content"] for item in all_candidates]
    co = cohere.Client(config.COHERE_API_KEY)
    
    rerank_response = co.rerank(
        model="rerank-english-v3.0",
        query=query, # Original query!
        documents=docs_to_rerank,
        top_n=dynamic_k,
    )

    # 4. SELECT TOP K
    retrieved: List[Document] = []
    for result in rerank_response.results:
        best_chunk = all_candidates[result.index]
        retrieved.append(
            Document(
                page_content=best_chunk["content"],
                metadata=best_chunk.get("metadata", {}),
            )
        )

    print(f"  → Final top {len(retrieved)} chunks selected via Cohere.")
    return retrieved


def generate_answer(chunks: List[Document], query: str) -> str:
    """
    Build a multimodal prompt from retrieved chunks and call the LLM.
    Text and table HTML go into the text block; images are appended as
    vision blocks so models that support vision can use them.
    """
    if not chunks:
        return "No relevant documents were found for your query."

    llm = _build_llm()

    # --- UPDATED STRICT SYSTEM PROMPT ---
    prompt = f"""You are a highly intelligent, direct, and concise AI enterprise assistant.
Answer the user's question using ONLY the provided context below.

CRITICAL RULES:
1. Be direct and concise.
2. DO NOT use phrases like "Based on Document 1" or "According to the text".
3. If the question has multiple parts, answer all of them clearly.
4. If a specific part of the question is missing from the context, only then say you don't have that specific info.
5. If the ENTIRE context is irrelevant, say: "I'm sorry, I don't have that information."

QUESTION: {query}

CONTEXT:
"""
    all_images: List[str] = []

    for i, chunk in enumerate(chunks, 1):
        prompt += f"--- Document {i} ---\n"

        raw = chunk.metadata.get("original_content")
        if raw:
            try:
                original = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                original = {}
        else:
            # Fallback: metadata may already be a dict (older ingestion runs)
            original = chunk.metadata if isinstance(chunk.metadata, dict) else {}

        raw_text = original.get("raw_text", chunk.page_content)
        if raw_text:
            prompt += f"TEXT:\n{raw_text}\n\n"

        for j, tbl in enumerate(original.get("tables_html", []), 1):
            prompt += f"TABLE {j}:\n{tbl}\n\n"

        all_images.extend(original.get("images_base64", []))

    prompt += "\nANSWER:"

    message_content = [{"type": "text", "text": prompt}]
    for img_b64 in all_images:
        message_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        })
    # --- THE FIX: Choose the model dynamically based on image presence ---
    needs_vision = len(all_images) > 0
    llm = _build_llm(needs_vision=needs_vision)

    try:
        response = llm.invoke([HumanMessage(content=message_content)])
        return response.content
    except Exception as exc:
        return f"Error generating answer: {exc}"


def run_query(query: str, k: int = 3) -> str:
    """Retrieve + generate: the full RAG query cycle."""
    chunks = retrieve_chunks(query, k=k)
    answer = generate_answer(chunks, query)
    return answer


# =========================================================================== #
#  ENTRY POINT                                                                 #
# =========================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal RAG pipeline")
    parser.add_argument("--ingest", action="store_true", help="Run ingestion")
    parser.add_argument("--pdf", type=str, help="Path to PDF (required with --ingest)")
    parser.add_argument("--query", type=str, help="Query to run against stored data")
    parser.add_argument("--source", type=str, default=None, help="Optional: Filter query by specific PDF filename")
    parser.add_argument("--export", action="store_true", help="Export chunks to JSON during ingestion")
    parser.add_argument("--k", type=int, default=3, help="Number of chunks to retrieve (default: 3)")
    args = parser.parse_args()

    if args.ingest:
        if not args.pdf:
            parser.error("--pdf is required when using --ingest")
        run_ingestion(args.pdf, export_json=args.export)

    if args.query:
        chunks = retrieve_chunks(args.query, k=args.k, source_file=args.source)
        answer = generate_answer(chunks, args.query)
        print(f"\n💬 Answer:\n{answer}")

    if not args.ingest and not args.query:
        # Default demo — comment out or change as needed
        demo_query = "What is the dimensionality (dmodel) used in the base Transformer model?"
        answer = run_query(demo_query)
        print(f"\n💬 Answer:\n{answer}")