-- ============================================================
-- RAG Pipeline — Supabase SQL Setup (v4 — full production)
-- Run this entire file in: Dashboard → SQL Editor → New Query
-- ============================================================
--
-- WHAT'S IN THIS FILE:
--   1. Extensions
--   2. Tables (documents + ingested_files registry)
--   3. Indexes (HNSW via halfvec, FTS, GIN metadata, file_hash)
--   4. Drop old function signatures
--   5. match_documents      — semantic search (LangChain internal)
--   6. hybrid_search        — semantic + keyword blended
--   7. get_document_types   — efficient DISTINCT taxonomy query
--   8. Supabase Storage setup note
--   9. Smoke tests
--
-- WHY halfvec:
--   pgvector HNSW caps at 2000 dims. nvidia/llama-nemotron outputs 2048.
--   Casting to halfvec(2048) at index time bypasses the limit.
--   Full float32 precision is still stored in the column.
-- ============================================================


-- ============================================================
-- 1. EXTENSIONS
-- ============================================================
create extension if not exists vector;
create extension if not exists pg_trgm;


-- ============================================================
-- 2. TABLES
-- ============================================================

-- Main vector store (LangChain SupabaseVectorStore format)
create table if not exists documents (
    id        uuid  primary key default gen_random_uuid(),
    content   text,
    metadata  jsonb,
    embedding vector(2048)
);

-- FIX: Dedicated file registry for fast O(1) duplicate checks.
-- is_file_already_ingested() used to do a JSONB containment scan on the
-- documents table. This table makes it an indexed lookup on a text column.
create table if not exists ingested_files (
    id            uuid        primary key default gen_random_uuid(),
    file_hash     text        unique not null,
    filename      text        not null,
    document_type text,
    chunk_count   int         default 0,
    ingested_at   timestamptz default now()
);


-- ============================================================
-- 3. INDEXES
-- ============================================================

-- Drop stale indexes from previous attempts
drop index if exists documents_embedding_idx;
drop index if exists documents_embedding_hnsw_idx;

-- HNSW via halfvec — bypasses the 2000-dim hard cap
create index if not exists documents_embedding_hnsw_idx
    on documents
    using hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
    with (m = 16, ef_construction = 64);

-- Full-text index on content (keyword leg of hybrid search)
create index if not exists documents_content_fts_idx
    on documents
    using gin (to_tsvector('english', content));

-- GIN on metadata (for @> containment filter queries)
create index if not exists documents_metadata_idx
    on documents
    using gin (metadata);

-- FIX: index on file_hash inside metadata for fast dedup checks
-- (used by is_file_already_ingested fallback path)
create index if not exists documents_metadata_filehash_idx
    on documents ((metadata->>'file_hash'));

-- Fast lookup on the file registry
create index if not exists ingested_files_hash_idx
    on ingested_files (file_hash);


-- ============================================================
-- 4. DROP ALL OLD FUNCTION SIGNATURES
-- ============================================================
drop function if exists match_documents(vector, int, jsonb)      cascade;
drop function if exists match_documents(vector, int, float, jsonb) cascade;
drop function if exists match_documents                            cascade;

drop function if exists hybrid_search(text, vector, int, jsonb)             cascade;
drop function if exists hybrid_search(text, vector, int, jsonb, float, float) cascade;
drop function if exists hybrid_search                                         cascade;

drop function if exists get_document_types() cascade;


-- ============================================================
-- 5. match_documents — pure semantic search
--    Used internally by LangChain's SupabaseVectorStore.
-- ============================================================
create or replace function match_documents(
    query_embedding  vector,
    match_count      int     default 5,
    filter           jsonb   default '{}'::jsonb
)
returns table (
    id          uuid,
    content     text,
    metadata    jsonb,
    similarity  float
)
language plpgsql
as $$
begin
    return query
    select
        d.id,
        d.content,
        d.metadata,
        (1 - (d.embedding::halfvec(2048) <=> query_embedding::halfvec(2048)))::float as similarity
    from documents d
    where (filter = '{}'::jsonb or d.metadata @> filter::jsonb)
    order by d.embedding::halfvec(2048) <=> query_embedding::halfvec(2048)
    limit match_count;
end;
$$;


-- ============================================================
-- 6. hybrid_search — semantic + keyword blended
--    Called by retrieve_chunks() in cl.py.
-- ============================================================
create or replace function hybrid_search(
    query_text       text,
    query_embedding  vector,
    match_count      int     default 10,
    filter           jsonb   default '{}'::jsonb,
    semantic_weight  float   default 0.7,
    keyword_weight   float   default 0.3
)
returns table (
    id             uuid,
    content        text,
    metadata       jsonb,
    combined_score float
)
language plpgsql
as $$
begin
    return query
    with
    semantic as (
        select
            d.id, d.content, d.metadata,
            (1 - (d.embedding::halfvec(2048) <=> query_embedding::halfvec(2048)))::float as score
        from documents d
        where (filter = '{}'::jsonb or d.metadata @> filter::jsonb)
        order by d.embedding::halfvec(2048) <=> query_embedding::halfvec(2048)
        limit match_count * 3
    ),
    keyword as (
        select
            d.id, d.content, d.metadata,
            ts_rank(
                to_tsvector('english', d.content),
                plainto_tsquery('english', query_text)
            )::float as raw_score
        from documents d
        where (filter = '{}'::jsonb or d.metadata @> filter::jsonb)
          and to_tsvector('english', d.content) @@ plainto_tsquery('english', query_text)
        order by raw_score desc
        limit match_count * 3
    ),
    keyword_norm as (
        select k.id, k.content, k.metadata,
            case
                when max(k.raw_score) over () = 0 then 0::float
                else (k.raw_score / max(k.raw_score) over ())::float
            end as score
        from keyword k
    ),
    blended as (
        select
            coalesce(s.id,       kn.id)       as id,
            coalesce(s.content,  kn.content)  as content,
            coalesce(s.metadata, kn.metadata) as metadata,
            (
                coalesce(s.score,  0::float) * semantic_weight +
                coalesce(kn.score, 0::float) * keyword_weight
            ) as combined_score
        from semantic s
        full outer join keyword_norm kn on s.id = kn.id
    )
    select b.id, b.content, b.metadata, b.combined_score
    from blended b
    order by b.combined_score desc
    limit match_count;
end;
$$;


-- ============================================================
-- 7. get_document_types — efficient taxonomy query
--    FIX: replaces the Python-side full-table-scan in
--    get_existing_categories(). Does DISTINCT server-side.
-- ============================================================
create or replace function get_document_types()
returns table (document_type text)
language sql
stable         -- marks it as read-only so the planner can cache it
as $$
    select distinct metadata->>'document_type' as document_type
    from documents
    where metadata->>'document_type' is not null
      and metadata->>'document_type' != 'unknown'
    order by 1;
$$;


-- ============================================================
-- 8. SUPABASE STORAGE — rag-images bucket
--
-- The Python code uploads extracted images to a bucket called
-- "rag-images" so they don't bloat the JSONB metadata column.
-- You must create this bucket manually in the Supabase dashboard:
--
--   Dashboard → Storage → New bucket
--   Name: rag-images
--   Public: YES  (so image URLs work without auth tokens)
--
-- The Python fallback stores base64 in metadata if the upload fails,
-- so ingestion won't crash even if the bucket doesn't exist yet.
-- ============================================================


-- ============================================================
-- 9. SMOKE TESTS — should return 0 rows, zero errors
-- ============================================================
select * from match_documents(
    query_embedding => array_fill(0, array[2048])::vector,
    match_count     => 3,
    filter          => '{}'::jsonb
);

select * from hybrid_search(
    query_text      => 'transformer model attention',
    query_embedding => array_fill(0, array[2048])::vector,
    match_count     => 3,
    filter          => '{}'::jsonb
);

select * from get_document_types();
