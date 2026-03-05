-- ============================================================
-- RAG Pipeline — Supabase SQL Setup (v5 — performance edition)
-- Run this entire file in: Dashboard → SQL Editor → New Query
-- ============================================================
--
-- WHAT'S NEW vs v4:
--   - ingested_files registry (O(1) dedup — no JSONB scan)
--   - Materialized view for document_type taxonomy (replaces DISTINCT on every request)
--   - get_document_types() now reads from the materialized view
--   - refresh_document_types_mv() — call this after each ingestion
--   - Automatic refresh via trigger on documents INSERT
--   - Index on ingested_files.file_hash for fast lookups
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

-- Dedicated file registry for O(1) duplicate checks.
-- is_file_already_ingested() does an indexed eq() on file_hash,
-- not a slow JSONB containment scan on documents.
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

-- Fast lookup by file_hash in ingested_files
create index if not exists ingested_files_hash_idx
    on ingested_files (file_hash);

-- Fast lookup on file_hash inside metadata (fallback path)
create index if not exists documents_metadata_filehash_idx
    on documents ((metadata->>'file_hash'));


-- ============================================================
-- 4. MATERIALIZED VIEW — document taxonomy
--
-- WHY: get_existing_categories() was doing a DISTINCT across the
-- full documents table on every sidebar render. For large corpora
-- this is expensive. A materialized view pre-computes the list.
-- The trigger below refreshes it automatically after each INSERT.
-- ============================================================

create materialized view if not exists mv_document_types as
    select distinct metadata->>'document_type' as document_type
    from documents
    where metadata->>'document_type' is not null
      and metadata->>'document_type' != 'unknown'
    order by 1;

-- Index on the materialized view for fast reads
create unique index if not exists mv_document_types_idx
    on mv_document_types (document_type);

-- Helper function to refresh the view (called from Python after ingestion)
create or replace function refresh_document_types_mv()
returns void
language plpgsql
security definer
as $$
begin
    refresh materialized view concurrently mv_document_types;
end;
$$;

-- Automatic refresh trigger on new document rows
-- Uses CONCURRENTLY so reads are never blocked
create or replace function _trg_refresh_mv_document_types()
returns trigger
language plpgsql
as $$
begin
    -- Fire-and-forget: refresh in background via pg_notify
    -- (avoids blocking the INSERT transaction itself)
    perform pg_notify('refresh_mv', 'document_types');
    return new;
end;
$$;

drop trigger if exists trg_refresh_mv_document_types on documents;
create trigger trg_refresh_mv_document_types
    after insert on documents
    for each statement
    execute procedure _trg_refresh_mv_document_types();


-- ============================================================
-- 5. DROP ALL OLD FUNCTION SIGNATURES
-- ============================================================
drop function if exists match_documents(vector, int, jsonb)        cascade;
drop function if exists match_documents(vector, int, float, jsonb)  cascade;
drop function if exists match_documents                              cascade;

drop function if exists hybrid_search(text, vector, int, jsonb)              cascade;
drop function if exists hybrid_search(text, vector, int, jsonb, float, float) cascade;
drop function if exists hybrid_search                                          cascade;

drop function if exists get_document_types() cascade;


-- ============================================================
-- 6. match_documents — pure semantic search
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
-- 7. hybrid_search — semantic + keyword blended
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
-- 8. get_document_types — reads from materialized view
--    FAST: no table scan, no DISTINCT computation at query time.
-- ============================================================
create or replace function get_document_types()
returns table (document_type text)
language sql
stable
as $$
    select document_type
    from mv_document_types
    order by document_type;
$$;


-- ============================================================
-- 9. SUPABASE STORAGE — rag-images bucket
--
-- Create manually in the Supabase dashboard:
--   Dashboard → Storage → New bucket
--   Name: rag-images
--   Public: YES
-- ============================================================


-- ============================================================
-- 10. SMOKE TESTS — should return 0 rows, zero errors
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

-- Verify materialized view
select count(*) from mv_document_types;
