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

import config
from cl import run_ingestion, retrieve_chunks, generate_answer, get_existing_categories
# FIX: import all three auth functions used in this file
from auth import verify_password, get_daily_password, verify_admin_key

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
    "ingestion_queue":   None,
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
        # FIX: verify_admin_key is now properly imported above
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
    st.stop()


# =========================================================================== #
#  FULL SIDEBAR — only after login                                             #
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
                    doc_type = run_ingestion(
                        path, export_json=False, force=False,
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

    # Drain queue each rerun — thread-safe communication pattern
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
        st.progress(step / total if total else 0,
                    text=f"Step {step}/{total}: {status}")
        st.caption("Large PDFs can take several minutes…")
        time.sleep(2)
        st.rerun()

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
        st.session_state.ingestion_result = None

    # FIX: live taxonomy from DB, NOT config.ALLOWED_CATEGORIES hardcoded list
    st.header("🎯 Search Filters")
    live_cats   = get_existing_categories()
    cat_options = ["All"] + live_cats
    current     = st.session_state.selected_category
    current_idx = cat_options.index(current) if current in cat_options else 0
    # FIX: store result in session_state so queries actually use it
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
#  MAIN CHAT INTERFACE                                                         #
# =========================================================================== #
st.title("🔮 Multimodal RAG Assistant")
st.markdown("Ask questions about your PDFs — text, tables, and charts all understood.")

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
                st.session_state.messages.append({"role": "assistant", "content": response})

                # if chunks:
                #     with st.expander("🔍 Retrieved Sources", expanded=False):
                #         for i, chunk in enumerate(chunks, 1):
                #             meta     = chunk.metadata
                #             original = meta.get("original_content")
                #             if isinstance(original, str):
                #                 try:
                #                     original = json.loads(original)
                #                 except Exception:
                #                     original = {}
                #             elif not isinstance(original, dict):
                #                 original = {}

                #             src   = meta.get("source", f"Document {i}")
                #             cidx  = meta.get("chunk_index", "?")
                #             dtype = meta.get("document_type", "unknown")

                #             st.markdown(
                #                 f"<div class='source-box'>"
                #                 f"<b>[Source {i}]</b> {src} &nbsp;·&nbsp; chunk {cidx}"
                #                 f" &nbsp;·&nbsp; <code>{dtype}</code></div>",
                #                 unsafe_allow_html=True,
                #             )
                #             # Show a snippet of the raw text for transparency
                #             raw_text = original.get("raw_text", chunk.page_content)
                #             if raw_text:
                #                 st.info(raw_text[:400] + ("…" if len(raw_text) > 400 else ""))

                #             tables = original.get("tables_html", [])
                #             if tables:
                #                 st.write("📊 **Tables:**")
                #                 for tbl in tables:
                #                     st.markdown(tbl, unsafe_allow_html=True)

                #             images_b64 = original.get("images_base64", [])
                #             if images_b64:
                #                 st.write("🖼️ **Images:**")
                #                 for img_str in images_b64:
                #                     # Use HTML/CSS to enforce a strict maximum height and perfect aspect ratio
                #                     img_html = f'''
                #                         <div style="text-align: center; margin-bottom: 15px; padding: 10px; background-color: #1e293b; border-radius: 8px;">
                #                             <img src="data:image/jpeg;base64,{img_str}" 
                #                                 style="max-width: 100%; max-height: 350px; object-fit: contain; border-radius: 4px;">
                #                         </div>
                #                     '''
                #                     st.markdown(img_html, unsafe_allow_html=True)
                #             st.divider()


                
                if chunks:
                    # 1. Collective Intelligence Indicator
                    unique_sources = list(set([doc.metadata.get("source", "Unknown") for doc in chunks]))
                    if len(unique_sources) > 1:
                        st.success(f"💡 This answer was synthesized from **{len(unique_sources)} different documents**.")
                    
                    with st.expander("🔍 Verified Source Materials", expanded=False):
                        # 2. Provenance Grouping (Organize by File)
                        from collections import defaultdict
                        grouped = defaultdict(list)
                        for chunk in chunks:
                            src = chunk.metadata.get("source", "Unknown")
                            grouped[src].append(chunk)

                        for src, source_chunks in grouped.items():
                            st.markdown(f"### 📄 {src}")
                            
                            for i, chunk in enumerate(source_chunks, 1):
                                meta = chunk.metadata
                                # Handle JSON string vs dict for original_content
                                original = meta.get("original_content", "{}")
                                if isinstance(original, str):
                                    try: original = json.loads(original)
                                    except: original = {}
                                
                                cidx = meta.get("chunk_index", "?")
                                dtype = meta.get("document_type", "unknown")

                                # Professional Citation Card
                                st.markdown(
                                    f"""<div style="border-left: 3px solid #10b981; padding: 10px; background: #1e293b; border-radius: 6px; margin-bottom: 10px;">
                                        <span style="color: #34d399; font-weight: bold;">[Ref {i}]</span> 
                                        <span style="color: #94a3b8; font-size: 0.85rem;">Chunk {cidx} • Category: {dtype}</span>
                                    </div>""", 
                                    unsafe_allow_html=True
                                )

                                # Text Snippet
                                raw_text = original.get("raw_text", chunk.page_content)
                                if raw_text:
                                    st.info(raw_text[:400] + ("…" if len(raw_text) > 400 else ""))

                                # Balanced Image Rendering
                                images_b64 = original.get("images_base64", [])
                                if images_b64:
                                    for img_str in images_b64:
                                        img_html = f'''
                                            <div style="text-align: center; margin-bottom: 15px; padding: 10px; background-color: #0f172a; border-radius: 8px; border: 1px solid #334155;">
                                                <img src="data:image/jpeg;base64,{img_str}" 
                                                     style="max-width: 100%; max-height: 350px; object-fit: contain; border-radius: 4px;">
                                            </div>
                                        '''
                                        st.markdown(img_html, unsafe_allow_html=True)
                            
                            st.divider() # Separate different PDFs
            except Exception as e:
                st.error(f"Failed to generate answer: {e}")
