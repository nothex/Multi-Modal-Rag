import streamlit as st
import os
import tempfile
import base64
import json

import config
from cl import run_ingestion, retrieve_chunks, generate_answer, get_existing_categories
from auth import verify_password, get_daily_password

st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="🤖",
    layout="wide",
)

st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #0e1117; color: white; }
    </style>
""", unsafe_allow_html=True)


# =========================================================================== #
#  SESSION STATE INIT                                                          #
# =========================================================================== #
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_category" not in st.session_state:
    st.session_state.selected_category = "All"


# =========================================================================== #
#  ADMIN PANEL — rendered BEFORE st.stop() so it's always visible             #
#  Admin can open this, grab today's code, and hand it to users               #
#  without needing to be logged in themselves.                                 #
# =========================================================================== #
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    with st.expander("🛠️ Admin Panel", expanded=False):
        admin_key = st.text_input("Master Admin Key", type="password", key="admin_key_input")
        if admin_key and admin_key == os.getenv("MASTER_ADMIN_KEY"):
            daily_code = get_daily_password()
            st.success("Admin Verified ✅")
            st.write("Today's Guest Code:")
            st.code(daily_code, language="text")
            st.caption("Share this code with users for 24-hour access.")


# =========================================================================== #
#  AUTHENTICATION GATE                                                         #
#  st.stop() halts ALL further rendering — that's why the admin panel         #
#  must be drawn above this block, not below it.                              #
# =========================================================================== #
if not st.session_state.authenticated:
    st.title("🔒 Secure RAG Access")
    st.markdown("Enter the daily access code to use the Multimodal RAG Pipeline.")

    guest_pw = st.text_input("Daily Access Code", type="password")
    if st.button("Unlock System", use_container_width=True):
        if verify_password(guest_pw):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid Code. Please contact the administrator.")
    st.stop()   # <-- nothing below here runs until authenticated


# =========================================================================== #
#  FULL SIDEBAR — only rendered after login                                    #
# =========================================================================== #
with st.sidebar:
    st.header("📄 Document Management")

    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

    if st.button("⚙️ Process Document", use_container_width=True):
        if uploaded_file:
            with st.spinner("Processing… (parsing, AI summaries, embedding — may take a few minutes)"):
                temp_dir = tempfile.gettempdir()
                real_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(real_file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                try:
                    run_ingestion(real_file_path, export_json=False)
                    st.success(f"✅ Successfully processed {uploaded_file.name}!")
                except ValueError as e:
                    st.warning(f"⚠️ Document rejected: {e}")
                except Exception as e:
                    st.error(f"❌ Ingestion failed: {e}")
                finally:
                    if os.path.exists(real_file_path):
                        os.remove(real_file_path)
        else:
            st.warning("Please upload a PDF first.")

    st.header("🎯 Search Filters")
    st.markdown("Narrow your search by document type.")
    # Dynamic taxonomy: pull live categories from the database
    live_categories = get_existing_categories()
    category_options = ["All"] + live_categories
    st.session_state.selected_category = st.selectbox(
        "Filter by Document Type:",
        options=category_options,
        index=0,
    )

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("""
### How it works
1. **Upload** your PDF.
2. **Process** it — extracts text, tables, and images.
3. **Chat** with your document using the Multimodal LLM!
""")


# =========================================================================== #
#  MAIN CHAT INTERFACE                                                         #
# =========================================================================== #
st.title("🔮 Multimodal RAG Assistant")
st.markdown("Ask complex questions about your PDFs — including data from tables and charts.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your document..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching and reading documents..."):
            try:
                active_category = st.session_state.get("selected_category", "All")
                chunks = retrieve_chunks(prompt, category=active_category)
                response = generate_answer(chunks, prompt)

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

                if chunks:
                    with st.expander("🔍 View Retrieved Sources & Images"):
                        for i, chunk in enumerate(chunks, 1):
                            st.markdown(f"**Source {i}**")

                            try:
                                metadata = json.loads(
                                    chunk.metadata.get("original_content", "{}")
                                )
                            except Exception:
                                metadata = {}

                            raw_text = metadata.get("raw_text", chunk.page_content)
                            st.info(raw_text[:300] + ("..." if len(raw_text) > 300 else ""))

                            tables = metadata.get("tables_html", [])
                            if tables:
                                st.write("📊 **Tables found in this chunk:**")
                                for tbl in tables:
                                    st.markdown(tbl, unsafe_allow_html=True)

                            images = metadata.get("images_base64", [])
                            if images:
                                st.write("🖼️ **Images found in this chunk:**")
                                for img_str in images:
                                    img_bytes = base64.b64decode(img_str)
                                    st.image(img_bytes, use_container_width=True)

                            st.divider()

            except Exception as e:
                st.error(f"Failed to generate answer: {e}")
