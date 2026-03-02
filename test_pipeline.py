"""
Quick smoke test — run with: python test_pipeline.py
Tests all major pipeline functions without needing a real PDF.
Exits with code 1 if any test fails.
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
    assert config.SUPABASE_URL,        "SUPABASE_URL missing from .env"
    assert config.OPENROUTER_API_KEY,  "OPENROUTER_API_KEY missing from .env"
    assert config.COHERE_API_KEY,      "COHERE_API_KEY missing from .env"
    assert config.EMBEDDING_MODEL,     "EMBEDDING_MODEL not set in config.py"
check("Config loads + all env vars present", test_config)

# ── 2. Supabase connection ────────────────────────────────────
def test_supabase():
    from cl import _build_supabase_client
    client = _build_supabase_client()
    result = client.table("documents").select("id").limit(1).execute()
    # Just checking no exception — result.data may legitimately be []
check("Supabase connection reachable", test_supabase)

# ── 3. Embedding model + correct dimensions ──────────────────
def test_embeddings():
    from cl import _build_embeddings
    emb = _build_embeddings()
    vec = emb.embed_query("test query")
    assert len(vec) == 2048, f"Expected 2048 dims, got {len(vec)}"
check("Embedding model returns 2048-dim vector", test_embeddings)

# ── 4. get_existing_categories returns a list ─────────────────
def test_categories():
    from cl import get_existing_categories
    cats = get_existing_categories()
    assert isinstance(cats, list), "Expected list"
    print(f"       (found {len(cats)} categories: {cats})")
check("get_existing_categories() returns list", test_categories)

# ── 5. _sanitize_category handles all edge cases ─────────────
def test_sanitize():
    from cl import _sanitize_category
    cases = {
        "Machine Learning Paper": "machine_learning_paper",
        "ML-paper!":              "ml_paper",
        "  ":                     "general_document",
        "":                       "general_document",
        "UPPER_CASE":             "upper_case",
        "already_good":           "already_good",
        "has  spaces":            "has_spaces",
        "123_numeric":            "123_numeric",
    }
    for raw, expected in cases.items():
        result = _sanitize_category(raw)
        assert result == expected, f"_sanitize_category({raw!r}) = {result!r}, expected {expected!r}"
check("_sanitize_category() all edge cases", test_sanitize)

# ── 6. get_file_fingerprint is deterministic ─────────────────
def test_fingerprint():
    from cl import get_file_fingerprint
    path = "/tmp/_rag_test_hash.txt"
    with open(path, "w") as f:
        f.write("hello world test content")
    h1 = get_file_fingerprint(path)
    h2 = get_file_fingerprint(path)
    assert h1 == h2,    "Hash not deterministic across two calls"
    assert len(h1) == 32, f"MD5 should be 32 hex chars, got {len(h1)}"
    # Change content → different hash
    with open(path, "w") as f:
        f.write("different content")
    h3 = get_file_fingerprint(path)
    assert h1 != h3, "Different content should produce different hash"
    os.remove(path)
check("get_file_fingerprint() deterministic + content-sensitive", test_fingerprint)

# ── 7. Pydantic schema accepts partial input ──────────────────
def test_schema_defaults():
    from cl import DocumentGraphMetadata
    # Only required field in practice is nothing — all have defaults
    m = DocumentGraphMetadata(is_allowed=True, document_type="test_doc")
    assert m.key_entities   == [],                    "key_entities should default to []"
    assert m.primary_topics == [],                    "primary_topics should default to []"
    assert m.brief_summary  == "No summary available.", "brief_summary default wrong"
check("DocumentGraphMetadata handles partial input gracefully", test_schema_defaults)

# ── 8. Schema absorbs unknown fields without crashing ─────────
def test_schema_extra_fields():
    from cl import DocumentGraphMetadata
    # Simulate older LLM response that includes 'categories' and 'audience'
    m = DocumentGraphMetadata(
        is_allowed=True,
        document_type="research_paper",
        categories=["AI", "NLP"],   # extra field from old schema
        audience="researchers",      # extra field from old schema
    )
    assert m.document_type == "research_paper"
    assert m.categories == ["AI", "NLP"]  # absorbed, not crashed
check("DocumentGraphMetadata absorbs extra fields (categories/audience)", test_schema_extra_fields)

# ── 9. model_construct() fallback never raises ────────────────
def test_schema_construct():
    from cl import DocumentGraphMetadata
    # This is used in the except block of extract_document_entities
    m = DocumentGraphMetadata.model_construct(
        is_allowed=True,
        document_type="general_document",
        key_entities=[],
        primary_topics=[],
        brief_summary="Classification failed.",
        categories=None,
        audience=None,
    )
    assert m.is_allowed == True
    assert m.document_type == "general_document"
check("DocumentGraphMetadata.model_construct() fallback works", test_schema_construct)

# ── 10. LLM builders return correct model names ───────────────
def test_llm_builder():
    from cl import _build_llm
    import config
    llm_cls  = _build_llm(use_classifier=True)
    llm_vis  = _build_llm(needs_vision=True)
    llm_txt  = _build_llm()
    # model attribute may be .model or .model_name depending on LangChain version
    def get_model(llm):
        return getattr(llm, "model", None) or getattr(llm, "model_name", None)
    assert get_model(llm_cls) == config.CLASSIFIER_LLM_MODEL, "Classifier model mismatch"
    assert get_model(llm_vis) == config.VISION_LLM_MODEL,     "Vision model mismatch"
    assert get_model(llm_txt) == config.TEXT_LLM_MODEL,       "Text model mismatch"
check("_build_llm() routes to correct model for each task", test_llm_builder)

# ── 11. generate_answer() with empty chunk list ───────────────
def test_empty_answer():
    from cl import generate_answer
    ans = generate_answer([], "any question")
    assert ans == "No relevant documents were found for your query."
check("generate_answer([]) returns graceful message", test_empty_answer)

# ── 12. retrieve_chunks() doesn't crash on sparse DB ─────────
def test_retrieval_no_crash():
    from cl import retrieve_chunks
    results = retrieve_chunks("zzz_this_query_matches_nothing_xkcd_42")
    assert isinstance(results, list), "Should return list, even if empty"
check("retrieve_chunks() returns list without crash on no-match query", test_retrieval_no_crash)

# ── 13. Duplicate ingestion check (hash-based) ───────────────
def test_is_already_ingested():
    from cl import is_file_already_ingested
    # This hash almost certainly doesn't exist in your DB
    result = is_file_already_ingested("00000000000000000000000000000000")
    assert result == False, "Should return False for a hash that doesn't exist"
check("is_file_already_ingested() returns False for unknown hash", test_is_already_ingested)

# ── 14. _separate_content handles chunk with no orig_elements ─
def test_separate_content_no_elements():
    from cl import _separate_content

    class FakeMetadata:
        pass  # no orig_elements attribute

    class FakeChunk:
        text = "Some plain text"
        metadata = FakeMetadata()

    result = _separate_content(FakeChunk())
    assert result["text"]   == "Some plain text"
    assert result["tables"] == []
    assert result["images"] == []
    assert "text" in result["types"]
check("_separate_content() handles chunk with no orig_elements", test_separate_content_no_elements)

# ── 15. JSON serialization of metadata doesn't break ─────────
def test_metadata_serialization():
    import json
    metadata = {
        "source": "test.pdf",
        "file_hash": "abc123",
        "document_type": "research_paper",
        "entities": ["BERT", "Transformer"],
        "topics": ["NLP", "deep learning"],
        "summary": "A paper about attention mechanisms.",
        "chunk_index": 1,
        "original_content": json.dumps({
            "raw_text": "Some text with unicode: café naïve",
            "tables_html": ["<table><tr><td>data</td></tr></table>"],
            "images_base64": [],
        })
    }
    # Round-trip through JSON
    serialized   = json.dumps(metadata, ensure_ascii=False)
    deserialized = json.loads(serialized)
    assert deserialized["source"] == "test.pdf"
    inner = json.loads(deserialized["original_content"])
    assert "café" in inner["raw_text"]
check("Metadata JSON round-trip with unicode", test_metadata_serialization)

# ── Summary ──────────────────────────────────────────────────
print()
print("=" * 60)
total = 15
passed = total - len(failures)
print(f"Results: {passed}/{total} passed")
if failures:
    print(f"\n❌ Failed tests:")
    for f in failures:
        print(f"   - {f}")
    sys.exit(1)
else:
    print("✅ All tests passed — pipeline is healthy.")
