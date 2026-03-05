"""
app.py — Streamlit UI for Multimodal RAG Pipeline

New in this version:
  - Document Manager tab: view, edit category, delete files
  - Category override on ingestion (accept or correct detected category)
  - All ingested files shown with title, category badge, chunk count
  - Edit category inline — pick existing or type new
  - Delete file removes all chunks + ingested_files row + centroid refresh
  - Moving a file between categories patches all its chunks atomically
"""
import os
import re
import tempfile
import json
import threading
import queue
import time

import streamlit as st

import config
from cl import (
    run_ingestion, retrieve_chunks, generate_answer,
    get_existing_categories, _apply_category_override, delete_document,
)
from auth import verify_password, get_daily_password, verify_admin_key

st.set_page_config(page_title="Multimodal RAG", page_icon="🔮", layout="wide")

st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #0e1117; color: white; }

    /* Uniform source cards */
    .source-card {
        border: 1px solid #1e293b;
        border-left: 4px solid #10b981;
        padding: 14px 16px;
        background: #0f172a;
        border-radius: 8px;
        margin-bottom: 12px;
        min-height: 80px;
    }
    .source-card.medium { border-left-color: #f59e0b; }
    .source-card.low    { border-left-color: #ef4444; }

    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-left: 6px;
    }
    .badge-high   { background: #064e3b; color: #34d399; }
    .badge-medium { background: #451a03; color: #fbbf24; }
    .badge-low    { background: #450a0a; color: #f87171; }
    .badge-crown  { background: #1e1b4b; color: #a5b4fc; }
    .badge-cat    { background: #1e293b; color: #94a3b8; font-size: 0.68rem; }

    .snippet {
        font-size: 0.88rem;
        color: #94a3b8;
        line-height: 1.55;
        margin-top: 8px;
        border-top: 1px solid #1e293b;
        padding-top: 8px;
    }
    .src-header {
        font-size: 1rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 18px 0 8px 0;
        padding-bottom: 4px;
        border-bottom: 1px solid #1e293b;
    }

    /* Document manager cards */
    .doc-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 10px;
    }
    .doc-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 4px;
    }
    .doc-meta {
        font-size: 0.78rem;
        color: #64748b;
    }
    </style>
""", unsafe_allow_html=True)


# =========================================================================== #
#  SESSION STATE                                                               #
# =========================================================================== #
defaults = {
    "authenticated":           False,
    "messages":                [],
    "selected_category":       "All",
    "ingestion_running":       False,
    "ingestion_status":        "",
    "ingestion_step":          0,
    "ingestion_total":         5,
    "ingestion_result":        None,
    "ingestion_queue":         None,
    "pending_category_review": None,
    "editing_file_hash":       None,   # which doc card is in edit mode
    "confirm_delete_hash":     None,   # which doc is pending delete confirm
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =========================================================================== #
#  ADMIN PANEL                                                                 #
# =========================================================================== #
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    with st.expander("🛠️ Admin Panel", expanded=False):
        admin_input = st.text_input("Master Admin Key", type="password", key="admin_key_input")
        if admin_input and verify_admin_key(admin_input):
            daily_code = get_daily_password()
            st.success("Admin Verified ✅")
            st.write("Today's Guest Code:")
            st.code(daily_code, language="text")
            st.caption("Resets automatically at midnight.")


# =========================================================================== #
#  AUTH GATE                                                                   #
# =========================================================================== #
if not st.session_state.authenticated:
    st.title("🔒 Secure RAG Access")
    st.markdown("Enter today's access code to continue.")
    guest_pw = st.text_input("Daily Access Code", type="password")
    if st.button("Unlock System", use_container_width=True):
        if verify_password(guest_pw):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid code. Contact the administrator for today's code.")
    st.stop()


# =========================================================================== #
#  SIDEBAR — upload + filter                                                   #
# =========================================================================== #
with st.sidebar:
    st.header("📄 Document Management")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if st.button("⚙️ Process Document", use_container_width=True):
        if not uploaded_file:
            st.warning("Please upload a PDF first.")
        elif st.session_state.ingestion_running:
            st.warning("Ingestion already in progress…")
        else:
            temp_dir  = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            q = queue.Queue()
            st.session_state.ingestion_queue   = q
            st.session_state.ingestion_running = True
            st.session_state.ingestion_status  = "Starting…"
            st.session_state.ingestion_step    = 0
            st.session_state.ingestion_result  = None

            def _run(path, q):
                try:
                    def _progress_cb(step, total, msg):
                        q.put(("progress", step, total, msg))
                    result = run_ingestion(
                        path, export_json=False, force=False,
                        progress_callback=_progress_cb,
                    )
                    q.put(("done", result))
                except Exception as exc:
                    q.put(("error", exc))
                finally:
                    if os.path.exists(path):
                        os.remove(path)

            threading.Thread(target=_run, args=(temp_path, q), daemon=True).start()
            st.rerun()

    # Drain ingestion queue
    if st.session_state.ingestion_running and st.session_state.ingestion_queue is not None:
        q = st.session_state.ingestion_queue
        while not q.empty():
            msg = q.get_nowait()
            if msg[0] == "progress":
                _, step, total, status = msg
                st.session_state.ingestion_step   = step
                st.session_state.ingestion_total  = total
                st.session_state.ingestion_status = status
            elif msg[0] in ("done", "error"):
                st.session_state.ingestion_result  = msg[1]
                st.session_state.ingestion_running = False
                st.session_state.ingestion_queue   = None
                break

        step   = st.session_state.ingestion_step
        total  = st.session_state.ingestion_total
        status = st.session_state.ingestion_status
        pct    = int((step / total) * 100) if total else 0
        st.progress(step / total if total else 0,
                    text=f"Step {step}/{total} ({pct}%): {status}")
        st.caption("Large PDFs with images can take several minutes…")
        time.sleep(2)
        st.rerun()

    # ── Ingestion result / category review ──
    if not st.session_state.ingestion_running and st.session_state.ingestion_result is not None:
        result = st.session_state.ingestion_result

        if isinstance(result, Exception):
            if isinstance(result, ValueError):
                st.warning(f"⚠️ {result}")
            else:
                st.error(f"❌ Ingestion failed: {result}")
            st.session_state.ingestion_result = None

        elif result == "already_ingested":
            st.info("⏭️ Already ingested — skipped duplicate.")
            st.session_state.ingestion_result = None

        elif isinstance(result, dict) and result.get("pending_review"):
            detected = result["document_type"]
            filename = result["filename"]
            live_cats = get_existing_categories()

            st.success(f"✅ **{filename}** processed!")
            st.markdown(f"**Detected category:** `{detected}`")
            st.caption("Accept or choose a different category:")

            options = (
                [detected]
                + [c for c in live_cats if c != detected]
                + ["＋ Type a new category"]
            )
            choice = st.selectbox("Category", options=options, key="cat_override_select")

            if choice == "＋ Type a new category":
                choice = st.text_input(
                    "New category (snake_case)", key="cat_override_input"
                )

            if st.button("✅ Confirm Category", use_container_width=True):
                if choice and choice.strip() and choice != "＋ Type a new category":
                    final_cat = re.sub(
                        r"[^a-z0-9]+", "_", choice.strip().lower()
                    ).strip("_")
                    _apply_category_override(result["file_hash"], final_cat)
                    st.session_state.ingestion_result = None
                    st.rerun()

    # ── Search filter ──
    st.header("🎯 Search Filters")
    live_cats   = get_existing_categories()
    cat_options = ["All"] + live_cats

    if not live_cats:
        st.caption("No documents ingested yet. Upload a PDF to get started!")

    current     = st.session_state.selected_category
    current_idx = cat_options.index(current) if current in cat_options else 0
    st.session_state.selected_category = st.selectbox(
        "Filter by Document Type:", options=cat_options, index=current_idx
    )

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("""
### How it works
1. **Upload** your PDF.
2. **Process** — OCR, AI summaries, embedding.
3. **Chat** with your documents!
""")
# =========================================================================== #
#  SOURCE RENDERER (used in chat tab)                                          #
# =========================================================================== #
def _render_sources(chunks):
    sorted_chunks = sorted(
        chunks,
        key=lambda c: c.metadata.get("relevance_score", 0.0),
        reverse=True,
    )
    unique_sources = list(
        dict.fromkeys(c.metadata.get("source", "Unknown") for c in sorted_chunks)
    )

    if len(unique_sources) > 1:
        st.success(
            f"💡 Answer synthesised from **{len(unique_sources)} different documents**."
        )

    with st.expander("🔍 Verified Source Materials", expanded=False):
        from collections import defaultdict
        grouped: dict = defaultdict(list)
        for chunk in sorted_chunks:
            grouped[chunk.metadata.get("source", "Unknown")].append(chunk)

        for src_idx, src in enumerate(unique_sources):
            source_chunks = grouped[src]
            doc_type = source_chunks[0].metadata.get("document_type", "unknown")

            st.markdown(
                f'<div class="src-header">📄 {src} '
                f'<span class="badge badge-crown" style="font-size:0.7rem;">{doc_type}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            for rank, chunk in enumerate(source_chunks):
                meta  = chunk.metadata
                score = meta.get("relevance_score")
                cidx  = meta.get("chunk_index", "?")
                total = meta.get("total_chunks", "?")
                pages = meta.get("page_numbers", [])
                is_top = src_idx == 0 and rank == 0

                original = meta.get("original_content", "{}")
                if isinstance(original, str):
                    try:
                        original = json.loads(original)
                    except Exception:
                        original = {}

                # Page string
                if pages:
                    page_str = f"p.{pages[0]}" if len(pages) == 1 else f"pp.{pages[0]}–{pages[-1]}"
                else:
                    page_str = ""

                # Badge
                crown = '<span class="badge badge-crown">👑 Top</span>' if is_top else ""
                if score is not None:
                    if score >= 0.7:
                        badge = f'{crown}<span class="badge badge-high">✅ {score:.0%}</span>'
                        card_cls = "source-card"
                    elif score >= 0.4:
                        badge = f'{crown}<span class="badge badge-medium">⚠️ {score:.0%}</span>'
                        card_cls = "source-card medium"
                    else:
                        badge = f'{crown}<span class="badge badge-low">🔴 {score:.0%}</span>'
                        card_cls = "source-card low"
                else:
                    badge, card_cls = crown, "source-card"

                page_display = f" • {page_str}" if page_str else ""
                st.markdown(
                    f'<div class="{card_cls}">'
                    f'  <div style="display:flex;justify-content:space-between;align-items:center;">'
                    f'    <span style="color:#64748b;font-size:0.8rem;">Chunk {cidx} of {total}{page_display}</span>'
                    f'    <span>{badge}</span>'
                    f'  </div>',
                    unsafe_allow_html=True,
                )

                raw_text = original.get("raw_text", chunk.page_content)
                if raw_text:
                    snippet = raw_text[:450] + ("…" if len(raw_text) > 450 else "")
                    st.markdown(
                        f'<div class="snippet">{snippet}</div></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown('</div>', unsafe_allow_html=True)

                images_b64 = original.get("images_base64", [])
                if images_b64:
                    with st.expander(
                        f"🖼️ View {len(images_b64)} image(s) from this chunk",
                        expanded=False,
                    ):
                        cols = st.columns(min(len(images_b64), 2))
                        for img_i, img_str in enumerate(images_b64):
                            with cols[img_i % 2]:
                                st.markdown(
                                    f'<div style="text-align:center;padding:8px;'
                                    f'background:#0f172a;border-radius:6px;'
                                    f'border:1px solid #334155;">'
                                    f'<img src="data:image/jpeg;base64,{img_str}" '
                                    f'style="max-width:100%;max-height:280px;'
                                    f'object-fit:contain;border-radius:4px;"></div>',
                                    unsafe_allow_html=True,
                                )

            st.divider()

# =========================================================================== #
#  MAIN TABS                                                                   #
# =========================================================================== #
tab_chat, tab_docs = st.tabs(["💬 Chat", "📁 Document Manager"])


# =========================================================================== #
#  TAB 1 — CHAT                                                                #
# =========================================================================== #
with tab_chat:
    st.title("🔮 Multimodal RAG Assistant")
    st.markdown("Ask questions about your PDFs — text, tables, and charts all understood.")

    if not st.session_state.messages:
        live = get_existing_categories()
        if not live:
            st.info("👋 No documents ingested yet. Upload a PDF in the sidebar to get started!")
        else:
            st.caption(
                f"📚 {len(live)} document categor{'y' if len(live)==1 else 'ies'} available: "
                + ", ".join(f"`{c}`" for c in live)
            )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about your documents…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching and reading documents…"):
                try:
                    active_category    = st.session_state.selected_category
                    history_for_memory = st.session_state.messages[:-1]

                    chunks   = retrieve_chunks(prompt, category=active_category)
                    response = generate_answer(chunks, prompt, chat_history=history_for_memory)

                    st.markdown(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

                    disclaimer_phrases = [
                        "i don't have that information",
                        "not provided in the context",
                        "i'm sorry",
                        "no information",
                    ]
                    is_valueless = any(p in response.lower() for p in disclaimer_phrases)

                    if chunks and not is_valueless:
                        _render_sources(chunks)
                    elif is_valueless:
                        st.warning("⚠️ No highly relevant documents found for this query.")

                except Exception as e:
                    st.error(f"Failed to generate answer: {e}")


# =========================================================================== #
#  TAB 2 — DOCUMENT MANAGER                                                   #
# =========================================================================== #
with tab_docs:
    st.title("📁 Document Manager")
    st.markdown("View, recategorise, or remove any ingested document.")

    # Load all ingested files from the registry
    from cl import _build_supabase_client
    supabase = _build_supabase_client()

    try:
        files_result = supabase.table("ingested_files") \
            .select("file_hash, filename, document_type, chunk_count, ingested_at") \
            .order("ingested_at", desc=True) \
            .execute()
        all_files = files_result.data or []
    except Exception as e:
        st.error(f"Could not load documents: {e}")
        all_files = []

    if not all_files:
        st.info("No documents ingested yet. Upload a PDF using the sidebar.")
        st.stop()

    # ── Group by category ──
    from collections import defaultdict
    by_category: dict = defaultdict(list)
    for f in all_files:
        cat = f.get("document_type") or "uncategorised"
        by_category[cat].append(f)

    live_cats_for_edit = get_existing_categories()

    st.markdown(
        f"**{len(all_files)} document{'s' if len(all_files)!=1 else ''} across "
        f"{len(by_category)} categor{'ies' if len(by_category)!=1 else 'y'}**"
    )
    st.divider()

    for cat, files in sorted(by_category.items()):
        # Category group header
        st.markdown(
            f'<div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">'
            f'  <span style="font-size:1.05rem; font-weight:700; color:#e2e8f0;">📂 {cat}</span>'
            f'  <span class="badge badge-cat">{len(files)} file{"s" if len(files)!=1 else ""}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        for f in files:
            fhash      = f["file_hash"]
            fname      = f["filename"]
            ftype      = f.get("document_type", "uncategorised")
            fchunks    = f.get("chunk_count", "?")
            fingested  = (f.get("ingested_at") or "")[:10]  # date only
            is_editing = st.session_state.editing_file_hash == fhash
            is_deleting = st.session_state.confirm_delete_hash == fhash

            # Card
            st.markdown(
                f'<div class="doc-card">'
                f'  <div class="doc-title">📄 {fname}</div>'
                f'  <div class="doc-meta">'
                f'    {fchunks} chunks &nbsp;·&nbsp; ingested {fingested}'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            col_edit, col_del, col_spacer = st.columns([1, 1, 4])

            with col_edit:
                if not is_editing and not is_deleting:
                    if st.button("✏️ Edit", key=f"edit_{fhash}"):
                        st.session_state.editing_file_hash   = fhash
                        st.session_state.confirm_delete_hash = None
                        st.rerun()

            with col_del:
                if not is_editing and not is_deleting:
                    if st.button("🗑️ Delete", key=f"del_{fhash}"):
                        st.session_state.confirm_delete_hash = fhash
                        st.session_state.editing_file_hash   = None
                        st.rerun()

            # ── EDIT MODE ──
            if is_editing:
                with st.container():
                    st.markdown(
                        f"**Editing category for:** `{fname}`  \n"
                        f"Current: `{ftype}`"
                    )
                    edit_options = (
                        [ftype]
                        + [c for c in live_cats_for_edit if c != ftype]
                        + ["＋ Type a new category"]
                    )
                    new_cat = st.selectbox(
                        "New category", options=edit_options,
                        key=f"editsel_{fhash}"
                    )
                    if new_cat == "＋ Type a new category":
                        new_cat = st.text_input(
                            "New category name (snake_case)",
                            key=f"editinput_{fhash}"
                        )

                    col_save, col_cancel = st.columns([1, 1])
                    with col_save:
                        if st.button("💾 Save", key=f"save_{fhash}"):
                            if (
                                new_cat
                                and new_cat.strip()
                                and new_cat != "＋ Type a new category"
                            ):
                                final = re.sub(
                                    r"[^a-z0-9]+", "_",
                                    new_cat.strip().lower()
                                ).strip("_")
                                if final != ftype:
                                    with st.spinner(f"Moving to `{final}`…"):
                                        _apply_category_override(fhash, final)
                                    st.success(f"✅ Moved to `{final}`")
                                else:
                                    st.info("Category unchanged.")
                                st.session_state.editing_file_hash = None
                                st.rerun()
                    with col_cancel:
                        if st.button("✖ Cancel", key=f"cancel_{fhash}"):
                            st.session_state.editing_file_hash = None
                            st.rerun()

            # ── DELETE CONFIRM MODE ──
            if is_deleting:
                st.warning(
                    f"⚠️ Delete **{fname}** and all its {fchunks} chunks? "
                    f"This cannot be undone."
                )
                col_yes, col_no = st.columns([1, 1])
                with col_yes:
                    if st.button("🗑️ Yes, delete", key=f"confirmyes_{fhash}"):
                        with st.spinner("Deleting…"):
                            delete_document(fhash)
                        st.success(f"✅ **{fname}** deleted.")
                        st.session_state.confirm_delete_hash = None
                        st.rerun()
                with col_no:
                    if st.button("✖ Cancel", key=f"confirmno_{fhash}"):
                        st.session_state.confirm_delete_hash = None
                        st.rerun()

        st.divider()
