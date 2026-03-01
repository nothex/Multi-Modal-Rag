# 🔮 Multimodal RAG Architect Pro

A production-grade, multi-domain Retrieval-Augmented Generation (RAG) pipeline designed to handle complex, unstructured PDF data across diverse industries (e.g., AI Research and Chemistry).

---

## 🚀 Key Architectural Features

### 1. The Semantic Bouncer (Graph Extraction)
- Uses an LLM-based classifier during ingestion to extract **Graph Metadata**.
- Automatically identifies `document_type` and key entities.
- Enables **Hard-Filtering** at the database level to prevent context bleed between unrelated domains.

### 2. Multi-Query Rewriter & Decomposition
- Solves the **Compound Query Problem**.
- Decomposes complex user prompts into multiple optimized sub-queries.
- Ensures questions spanning multiple topics (e.g., *Machine Learning authors* and *Atomic Weights*) retrieve relevant data for both.

### 3. Dynamic Context Windowing (Dynamic-K)
- Intelligently scales retrieval depth:
  - Lean `k=3` for simple questions (token-efficient).
  - Expands to `k=6` when multiple search intents are detected.
- Prevents loss of relevant data during reranking.

### 4. Cloud-Native Hybrid Search & Reranking
- **Vector Search:** Powered by NVIDIA Nemotron Embeddings (2048-dim).
- **Keyword Search:** BM25 via Supabase/Postgres full-text search.
- **Reranking:** Cohere English-v3 Cross-Encoder ensures only the highest-relevance chunks reach the LLM.

### 5. Multimodal Resilience
- **Rich Parsing:** Handles Text, HTML Tables, and Images using Unstructured.
- **Graceful Fallbacks:** Retry logic and fallbacks for Vision LLM outages (e.g., 502 errors), ensuring pipeline continuity.

---

## 🛠️ Tech Stack
- **Frontend:** Streamlit  
- **Orchestration:** LangChain / Python  
- **Database:** Supabase (PostgreSQL + pgvector)  
- **Models:**  
  - Text: Arcee-AI Trinity  
  - Vision/Extraction: NVIDIA Nemotron-VL  
  - Reranking: Cohere v3  
  - Parsing: Unstructured.io  

---

## 🏗️ Getting Started

1. **Clone the Repo**
   ```bash
   git clone <your-repo-link>
   ```
2. **Install Dependencies:**  
  ```bash
   pip install -r requirements.txt
  ```
3. **System Dependencies:**  
  Ensure ```poppler-utils``` and ```tesseract-ocr``` are installed on your OS.

4. **Environment Variables**
  - Setup the following in a ```.env``` file:
    - ```OPENROUTER_API_KEY```
    - ```SUPABASE_URL```
    - ```SUPABASE_SERVICE_KEY```
    - ```COHERE_API_KEY```

5. **Run App**

  ```bash
   streamlit run app.py
  ```


  # Multi-Modal-Rag
