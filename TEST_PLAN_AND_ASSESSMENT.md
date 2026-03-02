# RAG Pipeline — Test Plan & Industry Assessment

---

## PART 1 — EDGE CASE TEST MATRIX

### 1A. PDF Test Files to Download (All Free)

| Test Goal | PDF to Use | Where to Get It |
|---|---|---|
| Normal research paper | "Attention Is All You Need" | arxiv.org/abs/1706.03762 |
| Dense tables | "BERT" paper | arxiv.org/abs/1810.04805 |
| Heavy images/charts | "GPT-4 Technical Report" | arxiv.org/abs/2303.08774 |
| Pure scanned image (no text layer) | Any scanned book page | archive.org |
| Multi-language (Arabic/Chinese) | Any bilingual UN report | un.org/en/documents |
| Tiny file (<1 page) | Any 1-page abstract | arxiv.org any paper |
| Massive file (100+ pages) | "Deep Learning" textbook | maths.ed.ac.uk/~cmtan/BookChapters |
| Password-protected PDF | Create one in Word → Save as PDF with password | locally |
| Corrupted PDF | Rename a .jpg to .pdf | locally |
| Table-heavy spreadsheet-style | Any government statistical report | data.gov |
| Mixed: text + tables + images | "ResNet" paper | arxiv.org/abs/1512.03385 |
| Duplicate upload (same file twice) | Any PDF you already ingested | your test folder |
| Same content, different filename | Copy + rename any ingested PDF | your test folder |
| Non-English (Hindi/Urdu) | Any Hindi news article PDF | — |

---

### 1B. Ingestion Edge Cases

#### TEST-I-01: Empty / Blank PDF
```
What to test : Upload a PDF with 0 pages or all blank pages
How to create: python -c "from fpdf import FPDF; p=FPDF(); p.add_page(); p.output('blank.pdf')"
Expected     : is_allowed=False, clean error message in UI "empty or unreadable"
Watch for    : partition_pdf returning 0 elements — the elements[:25] loop
               produces empty sample_text, LLM may hallucinate is_allowed=True
Current gap  : NO explicit check for len(elements) == 0 before calling classifier
```

#### TEST-I-02: Password-Protected PDF
```
What to test : PDF that requires a password to open
Expected     : unstructured raises PasswordError or returns 0 elements
Watch for    : Silent failure — partition_pdf may return [] with no exception
Current gap  : No check for this; pipeline would ingest 0 chunks silently
```

#### TEST-I-03: Corrupted PDF (renamed image/docx)
```
What to test : A .jpg renamed to .pdf, or a truncated PDF
Expected     : partition_pdf raises, caught by run_ingestion try/except in app.py
Watch for    : The exception bubbles up as a generic "Ingestion failed" with no
               useful message for the user
```

#### TEST-I-04: Duplicate Upload (Same File)
```
What to test : Upload the exact same PDF twice without --force
Expected     : "SKIPPING: File already ingested" message shown in UI
Watch for    : is_file_already_ingested() pulls ALL metadata rows — on a large
               database this is a full table scan with no index. Fine for now,
               but will slow down at 10k+ rows.
```

#### TEST-I-05: Same Content, Different Filename
```
What to test : Copy "paper.pdf" → "paper_copy.pdf", upload both
Expected     : Second upload should be SKIPPED (hash match, not name match)
This should  : PASS — you use MD5 hash, not filename. Good design.
```

#### TEST-I-06: Scanned PDF (Image-Only, No Text Layer)
```
What to test : A PDF that is just a photograph of a document
Expected     : unstructured hi_res strategy runs OCR via tesseract
Watch for    : tesseract must be installed. If not: elements have empty .text
               → sample_text = "" → LLM gets blank prompt → unpredictable
Required     : pip install pytesseract + apt install tesseract-ocr
```

#### TEST-I-07: Massive PDF (200+ pages)
```
What to test : Large textbook or report
Expected     : Ingestion completes, many chunks created
Watch for    : Memory spike during partition_pdf hi_res (loads all pages at once)
               OpenRouter rate limits hit during _ai_summary loops
               Supabase upload times out if batching is not done
Current gap  : add_documents() sends all embeddings in one call — no batching
```

#### TEST-I-08: PDF with Only Images (Charts, No Text)
```
What to test : A PDF that is entirely charts/figures, like a slide deck exported
Expected     : Each chunk has images but near-empty text → _ai_summary runs
Watch for    : Vision model (qwen3-vl-235b) rate limit on many image chunks
               base64 strings stored in Supabase JSONB can hit row size limits
```

#### TEST-I-09: PDF with Special Characters / Equations
```
What to test : A math paper with LaTeX symbols, Greek letters, ∑∫∂
Expected     : Ingestion succeeds, symbols preserved in text
Watch for    : JSON serialization of metadata — some unicode can corrupt JSONB
```

#### TEST-I-10: Multi-Language PDF
```
What to test : PDF in Arabic, Chinese, or Hindi
Expected     : Classifier assigns appropriate category, text extracted
Watch for    : plainto_tsquery('english', ...) in hybrid_search SQL will fail
               silently on non-English text — keyword leg returns 0 results
Current gap  : Language is hardcoded as 'english' in the SQL functions
```

---

### 1C. Retrieval Edge Cases

#### TEST-R-01: Query That Matches Nothing
```
Query   : "What is the price of milk in 1987?"
Expected: "No relevant documents were found"
Watch for: Cohere reranker receiving an empty list → KeyError on result.index
Current : PASS — empty check before Cohere call exists
```

#### TEST-R-02: Very Long Query (>500 words)
```
Query   : Paste an entire paragraph as the question
Expected: Query rewriter breaks it into sub-queries, retrieval works
Watch for: Embedding model token limit for query_embedding
           OpenRouter may reject very long inputs
```

#### TEST-R-03: Gibberish / Nonsense Query
```
Query   : "asdfjkl qwerty zxcvbn 123456"
Expected: No chunks retrieved or irrelevant chunks, graceful "I don't have that info"
Watch for: Cohere crashing on reranking nonsense against nonsense chunks
```

#### TEST-R-04: SQL Injection in Query
```
Query   : "'; DROP TABLE documents; --"
Expected: Query treated as plain text, no database damage
This    : PASS — you use RPC parameters (parameterized), not string interpolation
```

#### TEST-R-05: Category Filter on Non-Existent Category
```
Action  : Manually type a category in the filter that doesn't exist in DB
Expected: 0 chunks returned, graceful "no results" message
Watch for: filter_dict passed to SQL with unknown document_type value
           should return empty, not crash — SQL handles this correctly
```

#### TEST-R-06: Query When Database is Empty
```
Action  : Run a query before any document has been ingested
Expected: "No relevant documents found"
Watch for: get_existing_categories() returns [] → selectbox shows only "All"
           This is fine. But Cohere call with empty docs_to_rerank will crash.
Current gap: The empty check before Cohere IS present. PASS.
```

#### TEST-R-07: Concurrent Users (Two People Querying Simultaneously)
```
Action  : Open two browser tabs, query at the same time
Expected: Both get correct independent answers
Watch for: Streamlit session state is per-user, so this should be fine
           But _build_supabase_client() creates a new connection per call —
           connection pool exhaustion under load is a risk
```

---

### 1D. Authentication Edge Cases

#### TEST-A-01: Expired Daily Code
```
Action  : Use yesterday's access code
Expected: "Invalid Code" error
```

#### TEST-A-02: Admin Panel with Wrong Key
```
Action  : Enter wrong master admin key
Expected: No code shown, no error — just silence (correct behaviour)
Watch for: Timing attack — comparing strings with == is not constant-time
           For production use hmac.compare_digest() instead
```

#### TEST-A-03: Refresh Page After Login
```
Action  : Log in, then press F5
Expected: Stay logged in (session_state persists across reruns in same tab)
Watch for: Hard refresh (Ctrl+Shift+R) or new tab → back to login screen
           This is correct Streamlit behaviour, not a bug
```

---

### 1E. UI / Integration Edge Cases

#### TEST-U-01: Upload Non-PDF File
```
Action  : The file_uploader has type=["pdf"] so this is blocked at UI level
Expected: Streamlit prevents non-PDF uploads
```

#### TEST-U-02: Click "Process Document" Without Uploading
```
Expected: "Please upload a PDF first" warning
This    : PASS — check exists
```

#### TEST-U-03: Network Drop During Ingestion
```
Action  : Disconnect internet mid-ingestion
Expected: Exception caught, temp file cleaned up in finally block
This    : PASS — finally block removes temp file regardless
```

#### TEST-U-04: Very Long Answer (LLM outputs 5000+ words)
```
Expected: Streamlit renders it fully with scrolling
Watch for: No truncation — but the chat history grows and slow re-renders
```

---

## PART 2 — QUICK TEST SCRIPT

Save as `test_pipeline.py` and run before every deployment:

```python
"""
Quick smoke test — run with: python test_pipeline.py
Tests all major pipeline functions without needing a real PDF.
"""
import os, sys, json, hashlib, uuid, re
sys.path.insert(0, ".")

print("=" * 60)
print("RAG Pipeline Smoke Tests")
print("=" * 60)

failures = []

def check(name, fn):
    try:
        fn()
        print(f"  ✅ {name}")
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        failures.append(name)

# ── 1. Config loads ──────────────────────────────────────────
def test_config():
    import config
    assert config.SUPABASE_URL, "SUPABASE_URL missing"
    assert config.OPENROUTER_API_KEY, "OPENROUTER_API_KEY missing"
    assert config.COHERE_API_KEY, "COHERE_API_KEY missing"
check("Config loads and env vars present", test_config)

# ── 2. Supabase connection ────────────────────────────────────
def test_supabase():
    from cl import _build_supabase_client
    client = _build_supabase_client()
    result = client.table("documents").select("id").limit(1).execute()
    # Just checking no exception — result.data may be []
check("Supabase connection", test_supabase)

# ── 3. Embedding model ───────────────────────────────────────
def test_embeddings():
    from cl import _build_embeddings
    emb = _build_embeddings()
    vec = emb.embed_query("test query")
    assert len(vec) == 2048, f"Expected 2048 dims, got {len(vec)}"
check("Embedding model (2048 dims)", test_embeddings)

# ── 4. Category fetch ─────────────────────────────────────────
def test_categories():
    from cl import get_existing_categories
    cats = get_existing_categories()
    assert isinstance(cats, list)
check("get_existing_categories() returns list", test_categories)

# ── 5. _sanitize_category ─────────────────────────────────────
def test_sanitize():
    from cl import _sanitize_category
    assert _sanitize_category("Machine Learning Paper") == "machine_learning_paper"
    assert _sanitize_category("ML-paper!") == "ml_paper"
    assert _sanitize_category("  ") == "general_document"
    assert _sanitize_category("résumé") == "r_sum"  # non-ascii stripped
check("_sanitize_category()", test_sanitize)

# ── 6. get_file_fingerprint ───────────────────────────────────
def test_fingerprint():
    from cl import get_file_fingerprint
    # Write a temp file, hash it, check determinism
    path = "/tmp/_test_hash.txt"
    with open(path, "w") as f: f.write("hello world")
    h1 = get_file_fingerprint(path)
    h2 = get_file_fingerprint(path)
    assert h1 == h2, "Hash not deterministic"
    assert len(h1) == 32, "MD5 should be 32 hex chars"
    os.remove(path)
check("get_file_fingerprint() deterministic", test_fingerprint)

# ── 7. Pydantic schema defaults ───────────────────────────────
def test_schema_defaults():
    from cl import DocumentGraphMetadata
    # Partial dict — should not raise
    m = DocumentGraphMetadata(is_allowed=True, document_type="test_doc")
    assert m.key_entities == []
    assert m.primary_topics == []
    assert m.brief_summary == "No summary available."
check("DocumentGraphMetadata handles partial input", test_schema_defaults)

# ── 8. Schema absorbs unknown fields ─────────────────────────
def test_schema_extra_fields():
    from cl import DocumentGraphMetadata
    # Older LLM response with 'categories' and 'audience' fields
    m = DocumentGraphMetadata(
        is_allowed=True,
        document_type="research_paper",
        categories=["AI", "NLP"],
        audience="researchers"
    )
    assert m.document_type == "research_paper"
check("DocumentGraphMetadata absorbs extra fields (categories/audience)", test_schema_extra_fields)

# ── 9. LLM builder ───────────────────────────────────────────
def test_llm_builder():
    from cl import _build_llm
    import config
    llm = _build_llm(use_classifier=True)
    assert llm.model_name == config.CLASSIFIER_LLM_MODEL or llm.model == config.CLASSIFIER_LLM_MODEL
check("_build_llm() builds correct model", test_llm_builder)

# ── 10. Retrieval on empty query ──────────────────────────────
def test_empty_retrieval():
    from cl import retrieve_chunks
    # This will hit the real DB but should not crash
    results = retrieve_chunks("test query about nothing specific zxcvbn")
    assert isinstance(results, list)
check("retrieve_chunks() returns list (no crash on sparse results)", test_empty_retrieval)

# ── 11. generate_answer with no chunks ───────────────────────
def test_empty_answer():
    from cl import generate_answer
    ans = generate_answer([], "any question")
    assert ans == "No relevant documents were found for your query."
check("generate_answer([]) returns graceful message", test_empty_answer)

# ── Summary ──────────────────────────────────────────────────
print()
print("=" * 60)
if failures:
    print(f"❌ {len(failures)} test(s) FAILED: {failures}")
    sys.exit(1)
else:
    print("✅ All tests passed.")
```

---

## PART 3 — INDUSTRY GRADE ASSESSMENT

### What You Have That IS Industry Grade ✅

| Feature | Why It's Good |
|---|---|
| Deterministic UUIDs via `uuid5(hash + chunk_index)` | Idempotent upserts — re-ingesting never duplicates data |
| MD5 file fingerprinting for dedup | Content-based, not name-based — robust |
| Hybrid search (semantic + BM25 keyword) | Better recall than pure vector search alone |
| Cohere reranking | Cross-encoder reranking is what production RAG systems use |
| Query rewriting into sub-queries | Handles multi-part questions properly |
| `halfvec` casting for HNSW index | Correct solution to pgvector's dimension limit |
| Dynamic taxonomy via live DB query | Self-organising, no manual maintenance |
| Multimodal ingestion (text + tables + images) | Most RAG systems skip tables and images entirely |
| Pydantic structured output with safe defaults | Defensive, won't crash on LLM variance |
| Temp file cleanup in `finally` block | No disk leaks |
| Auth gate with daily rotating codes | Reasonable for a personal/team deployment |

---

### What's Missing for True Production Grade ⚠️

#### CRITICAL GAPS

**1. No chunked/batched embedding upload**
```python
# CURRENT — sends ALL documents in one API call:
vector_store.add_documents(documents, ids=ids)

# If you have 200 chunks, this makes 200 embedding API calls sequentially.
# OpenRouter will rate-limit you. Fix:
BATCH_SIZE = 20
for i in range(0, len(documents), BATCH_SIZE):
    batch_docs = documents[i:i+BATCH_SIZE]
    batch_ids  = ids[i:i+BATCH_SIZE]
    vector_store.add_documents(batch_docs, ids=batch_ids)
    time.sleep(1)  # respect rate limits
```

**2. `get_existing_categories()` scans the entire table**
```python
# CURRENT — fetches ALL rows, extracts document_type client-side
# At 100k rows this will be slow and waste bandwidth

# FIX — use a SQL function or a materialized view:
# SELECT DISTINCT metadata->>'document_type' FROM documents;
# Or add this to supabase_functions.sql:
# CREATE OR REPLACE FUNCTION get_document_types()
# RETURNS TABLE(document_type text) AS $$
#   SELECT DISTINCT metadata->>'document_type' FROM documents
#   WHERE metadata->>'document_type' IS NOT NULL
# $$ LANGUAGE sql;
```

**3. No empty-document guard before classifier**
```python
# CURRENT — if PDF has 0 extractable elements, sample_text = ""
# LLM gets a blank prompt and may hallucinate is_allowed=True

# FIX — add to run_ingestion() after partition_document():
if not elements:
    raise ValueError("PDF appears to be blank or unreadable (0 elements extracted).")
```

**4. Metadata JSON stored as string inside JSONB**
```python
# CURRENT — original_content is json.dumps(...) stored as a string
# inside a JSONB column. It's JSONB-in-JSONB-as-string — can't be
# queried with Postgres operators.

# This means you can't do:
# SELECT * FROM documents WHERE metadata->'original_content'->>'raw_text' ILIKE '%term%'

# FIX — store original_content as a real nested JSONB object:
# "original_content": {       ← real dict, not json.dumps()
#     "raw_text": "...",
#     "tables_html": [...],
#     "images_base64": [...]
# }
# Then in generate_answer(), read it as chunk.metadata["original_content"]
# (already a dict, no json.loads() needed)
```

**5. No observability / logging**
```
Production systems need structured logs:
- How long did each ingestion step take?
- Which chunks triggered vision model calls?
- What was the Cohere rerank score for the top result?
- How many tokens did each LLM call consume?

Fix: Add Python logging with timestamps, or integrate Langfuse/LangSmith
     for full LLM call tracing (both have free tiers).
```

**6. base64 images stored in Supabase JSONB**
```
A single image chunk can be 500KB–2MB as base64.
At 100 ingested PDFs with images, your metadata column
can easily hit hundreds of MB, slowing down all queries.

Fix: Upload images to Supabase Storage (S3-compatible bucket),
     store only the public URL in metadata.
     Retrieve the image URL at display time instead.
```

---

### Architectural Improvements (Nice to Have) 🔧

**A. Add a proper document registry table**
```sql
-- Instead of checking the documents table for file_hash,
-- maintain a separate clean registry:
CREATE TABLE ingested_documents (
    id           uuid primary key default gen_random_uuid(),
    file_hash    text unique not null,
    filename     text,
    document_type text,
    chunk_count  int,
    ingested_at  timestamptz default now()
);
-- is_file_already_ingested() becomes a fast indexed lookup on file_hash
-- instead of a JSONB containment scan
```

**B. Async ingestion for large files**
```
Currently the Streamlit UI freezes for the entire ingestion duration.
For a 100-page PDF this is 5–15 minutes.

Fix: Use a task queue (Celery + Redis, or even just a background thread)
     and show a progress bar via st.progress().
     The user can keep using the chat while ingestion runs.
```

**C. Conversation memory**
```
Currently every query is stateless — the LLM doesn't know what was
asked before. The chat history is displayed but not sent to the LLM.

Fix: Pass the last N messages as context to generate_answer():
messages_context = "\n".join([
    f"{m['role'].upper()}: {m['content']}"
    for m in st.session_state.messages[-6:]  # last 3 turns
])
# Prepend to the prompt in generate_answer()
```

**D. Answer citations**
```
Currently the answer says "Be direct, don't reference Document X"
but users can't verify where the answer came from.

Better pattern: Return citations inline like [1][2],
with source filename and chunk_index shown below the answer.
This is what Perplexity, ChatPDF, and enterprise RAG systems do.
```

**E. Embedding cache**
```
If the same query is asked twice, you recompute the embedding.
Fix: Cache embeddings with functools.lru_cache or Redis.
```

---

### Summary Scorecard

| Dimension | Score | Notes |
|---|---|---|
| Core RAG correctness | 8/10 | Hybrid + rerank is solid |
| Resilience / error handling | 6/10 | Missing empty-doc guard, no batching |
| Scalability | 4/10 | Full table scans, no batching, images in JSONB |
| Observability | 2/10 | Only print() statements, no structured logs |
| UI/UX | 7/10 | Clean, auth works, source viewer is great |
| Security | 6/10 | String comparison for auth (use hmac.compare_digest) |
| Data model | 6/10 | JSON-in-JSONB-as-string is a design smell |
| **Overall** | **6.5/10** | **Solid prototype, needs hardening for production** |

The honest summary: **this is well above average for a personal/portfolio RAG project** — the hybrid search, Cohere reranking, multimodal pipeline, and dynamic taxonomy put it ahead of 90% of tutorial-level RAG systems. The gaps are real but they're all solvable and none of them will crash the system under normal use. Fix the batching and the empty-document guard first — those are the two most likely to cause problems day-to-day.
