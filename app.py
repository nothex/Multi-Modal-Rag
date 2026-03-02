"""
app.py — Streamlit UI for Multimodal RAG Pipeline
"""
import os
import tempfile
import base64
import json
import threading
import queue
import time

import streamlit as st
from auth import verify_password,get_daily_password

st.set_page_config(page_title="Multimodal RAG", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #0e1117; color: white; }
    .source-box { border-left: 3px solid #10b981; padding: 8px 12px;
                  background: #0f172a; border-radius: 4px; margin-bottom: 8px; }
    </style>
""", unsafe_allow_html=True)


# =========================================================================== #
#  SESSION STATE INIT                                                          #
# =========================================================================== #
defaults = {
    "authenticated":     False,
    "messages":          [],
    "selected_category": "All",
    "ingestion_running": False,
    "ingestion_status":  "",
    "ingestion_step":    0,
    "ingestion_total":   5,
    "ingestion_result":  None,
    "ingestion_queue":   None,   # queue.Queue instance, lives here so threads can write to it
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =========================================================================== #
#  ADMIN PANEL — always visible, even before login                            #
# =========================================================================== #
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    with st.expander("🛠️ Admin Panel", expanded=False):
        admin_input = st.text_input("Master Admin Key", type="password", key="admin_key_input")
        # FIX: use verify_admin_key() which uses hmac.compare_digest internally
        if admin_input and verify_admin_key(admin_input):
            daily_code = get_daily_password()
            st.success("Admin Verified ✅")
            st.write("Today's Guest Code:")
            st.code(daily_code, language="text")
            st.caption("Resets automatically at midnight.")


# =========================================================================== #
#  AUTHENTICATION GATE                                                         #
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
    st.stop()   # nothing below renders until authenticated


# =========================================================================== #
#  FULL SIDEBAR — only after login                                             #
# =========================================================================== #
with st.sidebar:
    st.header("📄 Document Management")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    # ── Ingestion with progress bar ───────────────────────────────────────
    # WHY QUEUE, NOT SESSION_STATE IN THREAD:
    # Streamlit session_state is NOT thread-safe. Writing to it from a
    # background thread is silently dropped, leaving the UI stuck forever.
    # The fix: the thread writes to a queue.Queue; the main thread drains
    # it on every rerun via st.rerun(). The queue lives in session_state
    # so it persists across reruns of the same session.
    if "ingestion_queue" not in st.session_state:
        st.session_state.ingestion_queue = None  # set when ingestion starts

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

            # Create a fresh queue for this ingestion run
            q = queue.Queue()
            st.session_state.ingestion_queue   = q
            st.session_state.ingestion_running = True
            st.session_state.ingestion_status  = "Starting…"
            st.session_state.ingestion_step    = 0
            st.session_state.ingestion_result  = None

            def _run(path, q):
                try:
                    def _progress_cb(step, total, msg):
                        # Thread-safe: write to queue, NOT to session_state
                        q.put(("progress", step, total, msg))

                    doc_type = run_ingestion(
                        path,
                        export_json=False,
                        force=False,
                        progress_callback=_progress_cb,
                    )
                    q.put(("done", doc_type))
                except Exception as exc:
                    q.put(("error", exc))
                finally:
                    if os.path.exists(path):
                        os.remove(path)

            threading.Thread(target=_run, args=(temp_path, q), daemon=True).start()
            st.rerun()

    # Drain the queue on every rerun while ingestion is active
    if st.session_state.ingestion_running and st.session_state.ingestion_queue is not None:
        q = st.session_state.ingestion_queue
        # Read all messages currently in the queue
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

        # Show progress bar
        step   = st.session_state.ingestion_step
        total  = st.session_state.ingestion_total
        status = st.session_state.ingestion_status
        st.progress(step / total if total else 0,
                    text=f"Step {step}/{total}: {status}")
        st.caption("Large PDFs can take several minutes…")

        # Keep polling — rerun every 2 seconds
        time.sleep(2)
        st.rerun()

    # Show result once done
    if not st.session_state.ingestion_running and st.session_state.ingestion_result is not None:
        result = st.session_state.ingestion_result
        if isinstance(result, Exception):
            if isinstance(result, ValueError):
                st.warning(f"⚠️ {result}")
            else:
                st.error(f"❌ Ingestion failed: {result}")
        elif result == "already_ingested":
            st.info("⏭️ Already ingested — skipped duplicate.")
        else:
            st.success(f"✅ Processed! Category: **{result}**")
        st.session_state.ingestion_result = None  # clear so it doesn't re-flash

    # --- THE NEW GRAPH FILTER WIDGET ---
    st.header("🎯 Search Filters")
    st.markdown("Use AI-extracted metadata to narrow your search.")
    
    # Create a list of options: "All" plus the categories from your config
    category_options = ["All"] + config.ALLOWED_CATEGORIES
    
    # The interactive dropdown
    selected_category = st.selectbox(
        "Filter by Document Type:", 
        options=category_options,
        index=0 # Defaults to "All"
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
#  MAIN CHAT INTERFACE                                                         #
# =========================================================================== #
st.title("🔮 Multimodal RAG Assistant")
st.markdown("Ask questions about your PDFs — text, tables, and charts all understood.")

# Replay history
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
                active_category = st.session_state.selected_category

                # FIX: pass chat history for conversation memory
                # Strip the current user message (last item) since it's the question
                history_for_memory = st.session_state.messages[:-1]

                chunks = retrieve_chunks(prompt, category=active_category)
                response = generate_answer(
                    chunks,
                    prompt,
                    chat_history=history_for_memory,
                )

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

                # ── Source viewer ─────────────────────────────────────────
                if chunks:
                    with st.expander("🔍 Retrieved Sources", expanded=False):
                        for i, chunk in enumerate(chunks, 1):
                            meta = chunk.metadata

                            # FIX: original_content is now a real dict — no json.loads needed
                            original = meta.get("original_content")
                            if isinstance(original, str):
                                try:
                                    original = json.loads(original)
                                except Exception:
                                    original = {}
                            elif not isinstance(original, dict):
                                original = {}

                            src   = meta.get("source", f"Document {i}")
                            cidx  = meta.get("chunk_index", "?")
                            dtype = meta.get("document_type", "unknown")

                            st.markdown(
                                f"<div class='source-box'>"
                                f"<b>[Source {i}]</b> {src} &nbsp;·&nbsp; chunk {cidx} &nbsp;·&nbsp; "
                                f"<code>{dtype}</code></div>",
                                unsafe_allow_html=True,
                            )

                            raw_text = original.get("raw_text", chunk.page_content)
                            if raw_text:
                                st.info(raw_text[:400] + ("…" if len(raw_text) > 400 else ""))

                            tables = original.get("tables_html", [])
                            if tables:
                                st.write("📊 **Tables:**")
                                for tbl in tables:
                                    st.markdown(tbl, unsafe_allow_html=True)

                            # FIX: display from Storage URL if available, else base64 fallback
                            image_urls = original.get("image_urls", [])
                            images_b64 = original.get("images_base64", [])

                            if image_urls or images_b64:
                                st.write("🖼️ **Images:**")
                            for url in image_urls:
                                st.image(url, use_container_width=True)
                            for img_str in images_b64:
                                img_bytes = base64.b64decode(img_str)
                                st.image(img_bytes, use_container_width=True)

                            st.divider()

            except Exception as e:
                st.error(f"Failed to generate answer: {e}")