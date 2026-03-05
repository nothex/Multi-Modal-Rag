"""
classifier.py — Hierarchical Ensemble Document Classifier
==========================================================

3-stage cascade:
  Stage 1: Embedding nearest-centroid  (cosine similarity, no API calls after warmup)
  Stage 2: Multi-signal ensemble       (embedding + TF-IDF keyword vote)
  Stage 3: LLM chain-of-thought        (only for genuinely novel document types)

Special: Sparse/tabular document detection routes to visual classification
before the normal pipeline (catches periodic tables, reference charts etc.)

Each stage only activates if the previous stage's confidence is below its threshold.
Centroid embeddings are persisted to Supabase so they survive restarts.

Usage (from cl.py):
    from classifier import DocumentClassifier
    clf = DocumentClassifier()
    result = clf.classify(sample_text, elements)
    # result.document_type  → "machine_learning_paper"
    # result.confidence     → 0.91
    # result.stage_used     → "centroid"
    # result.is_new_type    → False
"""

import os
import re
import json
import math
import hashlib
import logging
import threading
import time
import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from collections import Counter

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from supabase.client import create_client
from dotenv import load_dotenv

import config

load_dotenv()
log = logging.getLogger("rag_pipeline.classifier")


# =========================================================================== #
#  RESULT SCHEMA                                                               #
# =========================================================================== #

@dataclass
class ClassificationResult:
    document_type:  str
    confidence:     float
    stage_used:     str          # "centroid" | "ensemble" | "llm" | "visual" | "fallback"
    is_new_type:    bool
    brief_summary:  str               = "No summary available."
    key_entities:   List[str]         = field(default_factory=list)
    primary_topics: List[str]         = field(default_factory=list)
    is_allowed:     bool              = True
    runner_up:      Optional[str]     = None
    runner_up_conf: Optional[float]   = None


# =========================================================================== #
#  KEYWORD TAXONOMY                                                            #
# =========================================================================== #

KEYWORD_TAXONOMY: Dict[str, List[str]] = {
    "machine_learning_paper": [
        "attention", "transformer", "neural", "gradient", "backpropagation",
        "epoch", "loss", "accuracy", "dataset", "benchmark", "BLEU", "perplexity",
        "softmax", "embedding", "fine-tuning", "pre-training", "inference",
        "architecture", "layers", "weights", "optimizer", "regularization",
    ],
    "chemistry_reference": [
        "atomic", "element", "periodic", "proton", "neutron", "electron",
        "valence", "isotope", "molar", "compound", "molecule", "bond",
        "oxidation", "reduction", "electronegativity", "atomic weight",
        "symbol", "group", "period", "noble gas", "halogen",
    ],
    "legal_contract": [
        "hereinafter", "whereas", "indemnify", "liability", "jurisdiction",
        "party", "parties", "agreement", "clause", "termination", "breach",
        "arbitration", "governing law", "warranty", "confidential",
        "intellectual property", "force majeure", "consideration",
    ],
    "financial_report": [
        "revenue", "EBITDA", "earnings", "fiscal", "quarterly", "balance sheet",
        "cash flow", "equity", "dividend", "shareholder", "audit", "GAAP",
        "operating income", "net income", "liabilities", "assets", "EPS",
    ],
    "medical_guideline": [
        "patient", "clinical", "diagnosis", "treatment", "dosage", "efficacy",
        "adverse", "contraindication", "prognosis", "protocol", "randomized",
        "placebo", "cohort", "mortality", "morbidity", "symptom", "therapeutic",
    ],
    "research_paper": [
        "abstract", "introduction", "methodology", "conclusion", "references",
        "hypothesis", "experiment", "results", "discussion", "literature review",
        "statistical significance", "sample size", "citation", "peer-reviewed",
    ],
    "technical_manual": [
        "installation", "configuration", "troubleshooting", "specification",
        "diagram", "schematic", "procedure", "maintenance", "warning", "caution",
        "figure", "table", "appendix", "firmware", "hardware", "interface",
    ],
    "hr_policy": [
        "employee", "policy", "leave", "benefits", "conduct", "disciplinary",
        "grievance", "harassment", "onboarding", "performance review",
        "termination", "compensation", "workplace", "diversity", "inclusion",
    ],
    "reference_chart": [
        "table", "chart", "graph", "values", "units", "scale", "axis",
        "legend", "column", "row", "measurement", "range", "threshold",
    ],
}


# =========================================================================== #
#  MODULE-LEVEL HELPERS                                                        #
# =========================================================================== #

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _sanitize(raw: str) -> str:
    s = raw.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "general_document"


def _tfidf_scores(text: str, taxonomy: Dict[str, List[str]]) -> Dict[str, float]:
    text_lower = text.lower()
    n_cats = len(taxonomy)
    scores: Dict[str, float] = {}

    keyword_doc_freq: Dict[str, int] = Counter()
    for kws in taxonomy.values():
        seen: set = set()
        for kw in kws:
            if kw not in seen:
                keyword_doc_freq[kw] += 1
                seen.add(kw)

    for cat, keywords in taxonomy.items():
        score = 0.0
        for kw in keywords:
            if kw.lower() in text_lower:
                idf = math.log((n_cats + 1) / (keyword_doc_freq[kw] + 1)) + 1
                score += idf
        scores[cat] = score / len(keywords) if keywords else 0.0

    max_s = max(scores.values()) if scores else 1.0
    if max_s > 0:
        scores = {k: v / max_s for k, v in scores.items()}
    return scores


# =========================================================================== #
#  CENTROID STORE                                                              #
# =========================================================================== #

class CentroidStore:
    TABLE = "category_centroids"

    def __init__(self):
        self._cache: Dict[str, Dict] = {}
        self._lock   = threading.Lock()
        self._client = None
        self._load_from_db()

    def _get_client(self):
        if self._client is None:
            self._client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
        return self._client

    def _load_from_db(self):
        try:
            result = self._get_client().table(self.TABLE).select("*").execute()
            for row in (result.data or []):
                self._cache[row["document_type"]] = {
                    "vector": np.array(row["centroid_vector"], dtype=np.float32),
                    "count":  row["document_count"],
                }
            log.info("Centroid store loaded: %d categories", len(self._cache))
        except Exception as exc:
            log.warning("Could not load centroids from DB (%s). Using in-memory only.", exc)

    def get_all(self) -> Dict[str, Dict]:
        return dict(self._cache)

    def update(self, doc_type: str, new_vector: np.ndarray):
        with self._lock:
            if doc_type in self._cache:
                old = self._cache[doc_type]
                n   = old["count"]
                new_centroid = (old["vector"] * n + new_vector) / (n + 1)
                self._cache[doc_type] = {
                    "vector": new_centroid.astype(np.float32),
                    "count":  n + 1,
                }
            else:
                self._cache[doc_type] = {
                    "vector": new_vector.astype(np.float32),
                    "count":  1,
                }
            self._persist(doc_type)

    def _persist(self, doc_type: str):
        try:
            entry = self._cache[doc_type]
            self._get_client().table(self.TABLE).upsert({
                "document_type":   doc_type,
                "centroid_vector": entry["vector"].tolist(),
                "document_count":  entry["count"],
            }, on_conflict="document_type").execute()
        except Exception as exc:
            log.warning("Could not persist centroid for '%s': %s", doc_type, exc)


# =========================================================================== #
#  MAIN CLASSIFIER                                                             #
# =========================================================================== #

class DocumentClassifier:
    """
    Hierarchical classifier with sparse-document pre-check.

    Flow:
      0. Sparse/tabular pre-check  → _classify_visual()  (catches reference charts)
      1. Centroid nearest-neighbour  (threshold 0.72)
      2. Ensemble vote               (threshold 0.38)
      3. LLM chain-of-thought        (fallback, novel types only)
    """

    STAGE1_THRESHOLD = 0.72
    STAGE2_THRESHOLD = 0.38
    MIN_CORPUS_SIZE  = 1

    def __init__(self):
        self._centroids   = CentroidStore()
        self._embed_cache: Dict[str, np.ndarray] = {}
        self._embed_lock  = threading.Lock()
        self._last_stage  = "default"

    # ------------------------------------------------------------------ #
    #  PUBLIC API                                                          #
    # ------------------------------------------------------------------ #

    def classify(self, sample_text: str, elements: list = None) -> ClassificationResult:
        """Main entry point. Returns a ClassificationResult."""

        if not sample_text or not sample_text.strip():
            return ClassificationResult(
                document_type="general_document",
                confidence=0.0,
                stage_used="default",
                is_new_type=False,
                is_allowed=False,
            )

        # ── Sparse/tabular pre-check ──
        # Must run BEFORE embedding pipeline.
        # Catches periodic tables, reference grids, lookup charts where
        # extracted text is too short or repetitive for normal classifiers.
        words        = sample_text.split()
        unique_ratio = len(set(words)) / len(words) if words else 0
        is_sparse    = len(words) < 200 or unique_ratio > 0.85

        if is_sparse and elements:
            log.info(
                "Sparse document detected (words=%d, unique_ratio=%.2f) → visual classification",
                len(words), unique_ratio,
            )
            return self._classify_visual(elements)

        # ── Normal 3-stage pipeline ──
        text    = sample_text[:3000]
        doc_vec = self._embed(text)

        # Stage 1 — centroid nearest-neighbour
        stage1 = self._stage1_centroid(doc_vec)
        if stage1 and stage1[1] >= self.STAGE1_THRESHOLD:
            log.info("Stage 1 PASS: '%s' (conf=%.3f)", stage1[0], stage1[1])
            doc_type, confidence = stage1
            is_new = False
            runner_up, runner_up_conf = self._runner_up(doc_vec, exclude=doc_type)
        else:
            log.info("Stage 1 low conf (%.3f) → Stage 2", stage1[1] if stage1 else 0.0)

            # Stage 2 — ensemble vote
            stage2 = self._stage2_ensemble(doc_vec, text)
            if stage2 and stage2[1] >= self.STAGE2_THRESHOLD:
                log.info("Stage 2 PASS: '%s' (conf=%.3f)", stage2[0], stage2[1])
                doc_type, confidence = stage2
                is_new = doc_type not in self._centroids.get_all()
                runner_up, runner_up_conf = self._runner_up(doc_vec, exclude=doc_type)
            else:
                log.info("Stage 2 low conf (%.3f) → Stage 3 LLM", stage2[1] if stage2 else 0.0)

                # Stage 3 — LLM chain-of-thought
                doc_type, confidence = self._stage3_llm(text)
                is_new = doc_type not in self._centroids.get_all()
                runner_up, runner_up_conf = None, None

        doc_type = _sanitize(doc_type)

        # Extract rich metadata — always runs regardless of stage used
        meta = self._extract_metadata(text, doc_type)

        # Reinforce centroid with this document's vector
        self._centroids.update(doc_type, doc_vec)

        return ClassificationResult(
            document_type  = doc_type,
            confidence     = round(confidence, 4),
            stage_used     = self._last_stage,
            is_new_type    = is_new,
            brief_summary  = meta.get("summary", "No summary available."),
            key_entities   = meta.get("entities", []),
            primary_topics = meta.get("topics", []),
            is_allowed     = meta.get("is_allowed", True),
            runner_up      = runner_up,
            runner_up_conf = round(runner_up_conf, 4) if runner_up_conf else None,
        )

    def update_centroid(self, doc_type: str, sample_text: str):
        """Call after confirmed ingestion to reinforce the centroid."""
        vec = self._embed(sample_text[:2000])
        self._centroids.update(doc_type, vec)

    # ------------------------------------------------------------------ #
    #  SPARSE / VISUAL CLASSIFICATION                                      #
    # ------------------------------------------------------------------ #

    def _classify_visual(self, elements: list) -> ClassificationResult:
        """
        For sparse/tabular documents where text classifiers fail.
        Builds a structural fingerprint from element types and sends
        that to the LLM — no raw OCR text needed.
        """
        self._last_stage = "visual"

        el_types    = [type(el).__name__ for el in elements[:30]]
        type_counts = Counter(el_types)

        text_sample = " ".join(
            el.text for el in elements[:20]
            if hasattr(el, "text") and el.text
        )[:500]

        structure_desc = (
            f"Document structure analysis:\n"
            f"- Element type distribution: {dict(type_counts)}\n"
            f"- Total elements: {len(elements)}\n"
            f"- Has tables: {type_counts.get('Table', 0) > 0}\n"
            f"- Has images: {type_counts.get('Image', 0) > 0}\n"
            f"- Text sample (first 500 chars): {text_sample}\n"
        )

        prompt = (
            f"Based on this document structure fingerprint, classify the document type.\n\n"
            f"{structure_desc}\n"
            f"A document with many short text fragments, chemical symbols, numbers, and "
            f"table elements is a REFERENCE CHART not a research paper.\n"
            f"A document with long prose paragraphs and citations is an ACADEMIC PAPER.\n\n"
            f'Output ONLY: {{"document_type": "snake_case_label", "confidence": 0.0_to_1.0}}'
        )

        try:
            llm = ChatOpenAI(
                model=config.CLASSIFIER_LLM_MODEL,
                openai_api_key=config.OPENROUTER_API_KEY,
                openai_api_base=config.OPENROUTER_BASE_URL,
                temperature=0.0,
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            raw   = re.sub(r"```(?:json)?", "", response.content).strip().strip("`")
            match = re.search(r'\{.*?\}', raw, re.DOTALL)
            if match:
                data     = json.loads(match.group())
                doc_type = _sanitize(data.get("document_type", "reference_chart"))
                conf     = float(data.get("confidence", 0.7))
                log.info("Visual classification: '%s' (%.2f)", doc_type, conf)

                meta = self._extract_metadata(structure_desc, doc_type)
                self._centroids.update(doc_type, self._embed(structure_desc))

                return ClassificationResult(
                    document_type  = doc_type,
                    confidence     = conf,
                    stage_used     = "visual",
                    is_new_type    = doc_type not in self._centroids.get_all(),
                    brief_summary  = meta.get("summary", ""),
                    key_entities   = meta.get("entities", []),
                    primary_topics = meta.get("topics", []),
                    is_allowed     = True,
                )
        except Exception as exc:
            log.warning("Visual classification failed: %s", exc)

        # Hard fallback — sparse doc, LLM failed → safest guess
        return ClassificationResult(
            document_type="reference_chart",
            confidence=0.5,
            stage_used="fallback",
            is_new_type=False,
        )

    # ------------------------------------------------------------------ #
    #  STAGE 1 — CENTROID NEAREST-NEIGHBOUR                               #
    # ------------------------------------------------------------------ #

    def _stage1_centroid(self, doc_vec: np.ndarray) -> Optional[Tuple[str, float]]:
        centroids = self._centroids.get_all()
        if len(centroids) < self.MIN_CORPUS_SIZE:
            return None

        best_type, best_score = None, -1.0
        for cat, data in centroids.items():
            score = _cosine(doc_vec, data["vector"])
            if score > best_score:
                best_score = score
                best_type  = cat

        self._last_stage = "centroid"
        return (best_type, best_score) if best_type else None

    # ------------------------------------------------------------------ #
    #  STAGE 2 — ENSEMBLE VOTE                                            #
    # ------------------------------------------------------------------ #

    def _stage2_ensemble(
        self,
        doc_vec: np.ndarray,
        text: str,
    ) -> Optional[Tuple[str, float]]:
        self._last_stage = "ensemble"

        centroids = self._centroids.get_all()

        # Signal A: cosine similarity to known document centroids
        embed_scores: Dict[str, float] = {
            cat: _cosine(doc_vec, data["vector"])
            for cat, data in centroids.items()
        }

        # Signal B: cosine similarity to category label embeddings
        tax_embed_scores: Dict[str, float] = {
            cat: _cosine(doc_vec, self._embed(cat.replace("_", " ")))
            for cat in KEYWORD_TAXONOMY
        }

        # Signal C: TF-IDF keyword matching
        tfidf_scores = _tfidf_scores(text, KEYWORD_TAXONOMY)

        # Weighted ensemble: centroid 0.45, label-embed 0.30, tfidf 0.25
        all_cats = set(list(embed_scores.keys()) + list(KEYWORD_TAXONOMY.keys()))
        combined: Dict[str, float] = {
            cat: (
                0.45 * embed_scores.get(cat, 0.0) +
                0.30 * tax_embed_scores.get(cat, 0.0) +
                0.25 * tfidf_scores.get(cat, 0.0)
            )
            for cat in all_cats
        }

        if not combined:
            return None

        best_type = max(combined, key=combined.__getitem__)
        return (best_type, combined[best_type])

    # ------------------------------------------------------------------ #
    #  STAGE 3 — LLM CHAIN-OF-THOUGHT                                     #
    # ------------------------------------------------------------------ #

    def _stage3_llm(self, text: str) -> Tuple[str, float]:
        self._last_stage = "llm"
        existing = list(self._centroids.get_all().keys())

        existing_hint = (
            "EXISTING CATEGORIES IN THIS SYSTEM:\n" +
            "\n".join(f"  - {c}" for c in existing) +
            "\n\nReuse one of the above if it genuinely fits. "
            "Only invent a new label if none of them fit.\n"
        ) if existing else "No categories exist yet — invent an appropriate label.\n"

        prompt = f"""You are a document classification engine. Think step by step.

{existing_hint}

DOCUMENT EXCERPT:
{text[:2000]}

CRITICAL DISTINCTION — classify by FORMAT and STRUCTURE, not topic:
- A periodic table → "chemistry_reference" (it IS a reference chart, not a paper about chemistry)
- A transformer paper → "machine_learning_paper" (it IS a paper, written in prose with abstract/methods)
- A legal document → "legal_contract" (it has clauses, parties, signatures)
- A financial statement → "financial_report" (it has balance sheets, P&L tables)

REASONING STEPS:
1. What FORMAT is this? (reference chart, academic paper, manual, contract, report, policy)
2. What DOMAIN? (chemistry, ML, law, finance, medicine)
3. Does it have prose paragraphs + abstract + citations? → it's a paper
4. Does it have a grid/table of values with symbols/units? → it's a reference chart
5. Pick the most specific label. NEVER label a reference chart as a paper.

Output ONLY this JSON:
{{"document_type": "your_label_here", "confidence": 0.0_to_1.0, "reasoning": "one sentence about FORMAT not topic"}}"""

        llm = ChatOpenAI(
            model=config.CLASSIFIER_LLM_MODEL,
            openai_api_key=config.OPENROUTER_API_KEY,
            openai_api_base=config.OPENROUTER_BASE_URL,
            temperature=0.0,
        )

        for attempt in range(3):
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                raw   = response.content.strip()
                raw   = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
                match = re.search(r'\{.*?\}', raw, re.DOTALL)
                if match:
                    data       = json.loads(match.group())
                    doc_type   = _sanitize(data.get("document_type", "general_document"))
                    confidence = float(data.get("confidence", 0.6))
                    reasoning  = data.get("reasoning", "")
                    log.info("LLM classified as '%s' (conf=%.2f): %s", doc_type, confidence, reasoning)
                    return doc_type, confidence
            except Exception as exc:
                if "429" in str(exc):
                    wait = (attempt + 1) * 10
                    log.warning("Rate limited. Waiting %ds before retry %d/3...", wait, attempt + 1)
                    time.sleep(wait)
                else:
                    log.warning("LLM stage 3 failed: %s", exc)
                    break

        return "general_document", 0.3

    # ------------------------------------------------------------------ #
    #  METADATA EXTRACTION                                                 #
    # ------------------------------------------------------------------ #

    def _extract_metadata(self, text: str, doc_type: str) -> dict:
        """Extract entities, topics, summary, is_allowed. Always runs after classification."""
        prompt = f"""Extract metadata from this document excerpt. Output ONLY valid JSON, no markdown.

Document type already determined: {doc_type}

Required JSON fields:
- "is_allowed": true (false ONLY if text is gibberish/spam/empty)
- "summary": one sentence describing what this document is about
- "entities": list of up to 5 key names (people, orgs, algorithms, chemicals, etc.)
- "topics": list of 2-3 broad themes

DOCUMENT:
{text[:1500]}

JSON:"""

        try:
            llm = ChatOpenAI(
                model=config.CLASSIFIER_LLM_MODEL,
                openai_api_key=config.OPENROUTER_API_KEY,
                openai_api_base=config.OPENROUTER_BASE_URL,
                temperature=0.1,
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            raw   = response.content.strip()
            raw   = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
            match = re.search(r'\{.*?\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as exc:
            log.warning("Metadata extraction failed: %s", exc)

        return {
            "is_allowed": True,
            "summary":    f"A {doc_type.replace('_', ' ')} document.",
            "entities":   [],
            "topics":     [],
        }

    # ------------------------------------------------------------------ #
    #  HELPERS                                                             #
    # ------------------------------------------------------------------ #

    def _embed(self, text: str) -> np.ndarray:
        """Cached embedding — avoids re-embedding the same text twice."""
        key = hashlib.md5(text.encode()).hexdigest() if text else "empty"
        if key in self._embed_cache:
            return self._embed_cache[key]

        embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=config.OPENROUTER_API_KEY,
            openai_api_base=config.OPENROUTER_BASE_URL,
            check_embedding_ctx_length=False,
            model_kwargs={"encoding_format": "float"},
        )
        vec = np.array(embeddings.embed_query(text), dtype=np.float32)
        with self._embed_lock:
            if len(self._embed_cache) >= 512:
                oldest = next(iter(self._embed_cache))
                del self._embed_cache[oldest]
            self._embed_cache[key] = vec
        return vec

    def _runner_up(
        self,
        doc_vec: np.ndarray,
        exclude: str,
    ) -> Tuple[Optional[str], Optional[float]]:
        """Second-best centroid match — shown in logs for debugging."""
        centroids = self._centroids.get_all()
        best_type, best_score = None, -1.0
        for cat, data in centroids.items():
            if cat == exclude:
                continue
            score = _cosine(doc_vec, data["vector"])
            if score > best_score:
                best_score = score
                best_type  = cat
        return (best_type, best_score) if best_type else (None, None)