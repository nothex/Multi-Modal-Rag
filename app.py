import streamlit as st
import os
import tempfile
import config
# Import the engine you just built!
from cl import run_ingestion, run_query

# Getting Images too if present
import base64
import json
from cl import retrieve_chunks, generate_answer # We import these instead of run_query
# Adding security via login password
import streamlit as st
from auth import verify_password,get_daily_password

st.set_page_config(page_title="Multimodal RAG", page_icon="🤖", layout="wide")

# --- AUTHENTICATION LOGIC ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .stChatFloatingInputContainer { background-color: rgba(0,0,0,0); }
    /* Style the upload box */
    .st-emotion-cache-1c7n2ri { 
        border: 2px dashed #4CAF50; 
        border-radius: 10px; 
        padding: 20px; 
        background-color: #f9f9f9;
    }
    /* Style the sidebar */
    [data-testid="stSidebar"] {
        background-color: #0e1117;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


# This keeps your chat history alive across interactions
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: Document Upload & Management ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80) # Generic Doc Icon
    st.header("📄 Document Management")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    # The Ingestion Button
    if st.button("Process Document",width="stretch"):
        if uploaded_file :
            # We use a spinner so the user knows it's working
            with st.spinner("Processing document... This may take a few minutes (Parsing, AI Summaries, Embedding)."):
                # FIX: Save the file using its ACTUAL name so the Gatekeeper works!
                temp_dir = tempfile.gettempdir()
                real_file_path = os.path.join(temp_dir, uploaded_file.name)
                # Save the uploaded file temporarily so unstructured can read it
                # with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                #     tmp_file.write(uploaded_file.getvalue())
                #     tmp_path = tmp_file.name
                with open(real_file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                try:
                    # Call YOUR ingestion function
                    run_ingestion(real_file_path, export_json=False)
                    st.success(f"✅ Successfully processed & verified {uploaded_file.name}!")
                except Exception as e:
                    st.error(f"❌ Ingestion failed: {e}")
                finally:
                    # Clean up the temp file
                    if os.path.exists(real_file_path):
                        os.remove(real_file_path)
        else:
            st.warning("Please upload a PDF first.")

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

    if st.button("🗑️ Clear Chat History", width="stretch"):
        st.session_state.messages = []
        st.rerun()
            
    st.markdown("---")
    st.markdown("""
    ### How it works:
    1. **Upload** your PDF.
    2. **Process** it (extracts text, tables, images).
    3. **Chat** with your document using your Multimodal LLM!
    """)
    # --- ADMIN SECTION ---
    st.markdown("---")
    with st.expander("🛠️ Admin Panel"):
        admin_key = st.text_input("Master Admin Key", type="password")
        if admin_key == os.getenv("MASTER_ADMIN_KEY"):
            daily_code = get_daily_password()
            st.success("Admin Verified")
            st.write("Today's Guest Code:")
            st.code(daily_code, language="text")
            st.caption("Provide this code to users for 24-hour access.")
# --- Main Chat Interface ---
if not st.session_state.authenticated:
    st.title("🔒 Secure RAG Access")
    st.markdown("Enter the daily access code to use the Multimodal RAG Pipeline.")
    guest_pw = st.text_input("Daily Access Code", type="password")
    if st.button("Unlock System", width="stretch"):
        if verify_password(guest_pw):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid Code. Please contact the administrator.")
    st.stop()


# --- 5. MAIN CHAT INTERFACE (Only loads if authenticated) ---
st.title("🔮 Multimodal RAG Assistant")
st.markdown("Ask complex questions about your PDFs, including data from tables and charts.")


# 1. Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Accept user input
if prompt := st.chat_input("Ask about your document..."):
    
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching Supabase and reading documents..."):
            try:
                # 1. Retrieve the chunks manually
                chunks = retrieve_chunks(prompt,category=selected_category)
                
                # 2. Generate the text answer
                response = generate_answer(chunks, prompt)
                
                # 3. Display the text answer
                st.markdown(response)
                
                # 4. Save to history
                st.session_state.messages.append({"role": "assistant", "content": response})

                # --- THE NEW FEATURE: DISPLAY SOURCES & IMAGES ---
                if chunks:
                    with st.expander("🔍 View Retrieved Sources & Images"):
                        for i, chunk in enumerate(chunks, 1):
                            st.markdown(f"**Source {i}:**")
                            # Safely load the JSON metadata
                            try:
                                metadata = json.loads(chunk.metadata.get("original_content", "{}"))
                            except:
                                metadata = {}
                            
                            # Show a snippet of the text
                            raw_text = metadata.get("raw_text", chunk.page_content)
                            st.info(raw_text[:300] + "...") 
                            
                            # 2. NEW: Display HTML Tables if they exist!
                            tables = metadata.get("tables_html", [])
                            if tables:
                                st.write("📊 **Tables found in this chunk:**")
                                for tbl in tables:
                                    # We use unsafe_allow_html so Streamlit renders the actual table grid
                                    st.markdown(tbl, unsafe_allow_html=True)
                            
                            # 3. Display Images if they exist
                            images = metadata.get("images_base64", [])
                            if images:
                                st.write("🖼️ **Images found in this chunk:**")
                                for img_str in images:
                                    img_bytes = base64.b64decode(img_str)
                                    st.image(img_bytes, width="stretch")
                            
                            st.divider()
                
            except Exception as e:
                st.error(f"Failed to generate answer: {e}")