"""
warmup_classifier.py
====================
Run once (or any time you add new confirmed documents) to seed the
centroid store from existing Supabase data.

It will:
  - Read all distinct document_types from ingested_files
  - For each type, fetch the embeddings of all its chunks from documents
  - Compute the mean centroid vector
  - Upsert into category_centroids — skipping types already trained

Safe to re-run — already-trained categories are never overwritten.
Only genuinely new categories (not in category_centroids yet) get trained.

Usage:
    python warmup_classifier.py
"""

import numpy as np
import logging
from collections import defaultdict
from supabase.client import create_client
from dotenv import load_dotenv
import config

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("warmup")


def warmup():
    supabase = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)

    # Step 1 — find which categories already have centroids
    log.info("Loading existing centroids...")
    existing = supabase.table("category_centroids").select("document_type").execute()
    already_trained = {row["document_type"] for row in (existing.data or [])}
    log.info("Already trained: %s", already_trained or "none")

    # Step 2 — find all confirmed document types from ingested_files
    log.info("Reading confirmed document types from ingested_files...")
    files = supabase.table("ingested_files") \
        .select("document_type") \
        .execute()

    all_types = {
        row["document_type"] for row in (files.data or [])
        if row.get("document_type") and row["document_type"] != "general_document"
    }
    log.info("All confirmed types: %s", all_types)

    # Step 3 — only train types not already in centroid store
    to_train = all_types - already_trained
    if not to_train:
        log.info("Nothing to train — all categories already have centroids.")
        return

    log.info("Will train centroids for: %s", to_train)

    # Step 4 — for each new type, fetch all chunk embeddings and compute mean
    for doc_type in to_train:
        log.info("Processing '%s'...", doc_type)

        # Fetch all document rows for this type
        # Supabase returns embeddings as list of floats
        rows = supabase.table("documents") \
            .select("embedding") \
            .eq("metadata->>document_type", doc_type) \
            .execute()

        if not rows.data:
            log.warning("No chunks found for '%s' — skipping.", doc_type)
            continue

        # Filter out any null embeddings
        vectors = [
            np.array(row["embedding"], dtype=np.float32)
            for row in rows.data
            if row.get("embedding")
        ]

        if not vectors:
            log.warning("No valid embeddings for '%s' — skipping.", doc_type)
            continue

        # Compute mean centroid
        centroid = np.mean(vectors, axis=0).astype(np.float32)
        log.info(
            "  '%s': %d chunks → centroid shape %s",
            doc_type, len(vectors), centroid.shape,
        )

        # Step 5 — upsert into category_centroids
        supabase.table("category_centroids").upsert({
            "document_type":   doc_type,
            "centroid_vector": centroid.tolist(),
            "document_count":  len(vectors),
        }, on_conflict="document_type").execute()

        log.info("  ✅ Centroid saved for '%s'", doc_type)

    log.info("Warmup complete. Trained %d new categories.", len(to_train))


if __name__ == "__main__":
    warmup()