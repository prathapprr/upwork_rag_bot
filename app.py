import os
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import time
import streamlit as st

# Configure Streamlit page settings first to ensure custom page name and icon
st.set_page_config(
    page_title="ProAnalyst - Upwork Technical Support AI Bot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Ensure stdout uses UTF-8 to prevent any encoding errors
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# Auto-initialize vector database if it does not exist (essential for cloud deployments like Streamlit Community Cloud)
persist_db_dir = "./chroma_db"
if not os.path.exists(persist_db_dir) or not os.listdir(persist_db_dir):
    try:
        # Check if the source text reference exists, if not, write a warning
        if os.path.exists("upwork_api_reference.txt"):
            with st.spinner("First-time setup: Initializing Vector Database..."):
                from vector_store import create_vector_store
                with open("upwork_api_reference.txt", "r", encoding="utf-8") as f:
                    text = f.read()
                create_vector_store(text, persist_db_dir)
        else:
            st.error("Setup Error: 'upwork_api_reference.txt' is missing. Please run ingest.py locally first.")
    except Exception as e:
        st.error(f"Setup Error: Failed to automatically initialize vector database: {e}")

# Import retrieval and llm_chain logic after auto-initialization block
from retrieval import get_relevant_context
from llm_chain import generate_rag_response
from experimental_modes import render_physics_sandbox_page

# Inject Premium Custom CSS for Rich Aesthetics, Dark Mode, Typography, and Glassmorphism
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
    /* Global Background and Typography */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title and Headers styling */
    h1, h2, h3, p {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #38bdf8 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        padding-top: 1rem;
    }
    
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Custom Card Containers for Glassmorphism */
    .premium-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .answer-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(12px);
        border-left: 5px solid #a855f7;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 20px;
        margin-top: 10px;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.2);
    }
    
    /* Performance Stats Badges */
    .metric-container {
        display: flex;
        gap: 15px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    
    .metric-badge {
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: #38bdf8;
        padding: 6px 12px;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-flex;
        align-items: center;
    }
    
    .metric-badge-alt {
        background: rgba(168, 85, 247, 0.15);
        border: 1px solid rgba(168, 85, 247, 0.3);
        color: #c084fc;
        padding: 6px 12px;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-flex;
        align-items: center;
    }
    
    /* Code block styling */
    .source-block {
        font-family: 'JetBrains Mono', monospace;
        background-color: #0b0f19 !important;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        font-size: 0.9rem;
        color: #e2e8f0;
        white-space: pre-wrap;
    }
    
    .source-header {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;
        margin-bottom: 4px;
        display: flex;
        justify-content: space-between;
    }
    
    /* Custom styling for standard Streamlit expanders */
    .streamlit-expanderHeader {
        background-color: rgba(30, 41, 59, 0.3) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        color: #f8fafc !important;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-title">ProAnalyst</div>', unsafe_allow_html=True)

# Sidebar for Mode Toggle
st.sidebar.markdown("## 🛠️ App Configuration")
app_mode = st.sidebar.radio(
    "Select System Mode:",
    ["Upwork API Consultant", "Experimental Physics Sandbox"],
    index=0
)

st.sidebar.markdown("---")

if app_mode == "Upwork API Consultant":
    # ------------------ UPWORK API CONSULTANT MODE ------------------
    st.markdown('<div class="subtitle">🤖 Upwork Technical Support AI Bot (RAG)</div>', unsafe_allow_html=True)
    
    # Main Form Card
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.subheader("Submit API Consultation Query")
    st.write("Ask technical support questions regarding Upwork's GraphQL and OAuth 2.0 API guidelines. The bot utilizes semantic retrieval and hallucination guards to ensure non-hallucinated answers.")
    
    # Form text input
    with st.form(key="query_form"):
        user_query = st.text_area(
            label="Enter your developer query:",
            placeholder="e.g., How long is an OAuth access token valid for?",
            height=100
        )
        
        submit_button = st.form_submit_button(label="Analyze & Retrieve")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Execute RAG Sequence on Submit
    if submit_button:
        if not user_query.strip():
            st.warning("Please enter a valid query before submitting.")
        else:
            with st.spinner("Executing RAG Pipeline (retrieving documentation & generating response)..."):
                # Record start time precisely before retrieval & LLM pipeline
                t_start = time.time()
                
                # 1. Semantic Retrieval (k=3)
                try:
                    context_chunks = get_relevant_context(user_query)
                except Exception as e:
                    st.error(f"Retrieval Phase Error: {e}")
                    context_chunks = []
                
                # 2. LLM response generation with Hallucination Guard
                answer, api_latency = generate_rag_response(user_query, context_chunks)
                
                # Record stop time precisely after raw JSON response is received & parsed
                t_stop = time.time()
                total_latency = t_stop - t_start
                
            # Display Results
            st.success("Query Analysis Complete!")
            
            # 1. Answer Card
            st.markdown("### 💡 AI Response")
            st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)
            
            # 2. Performance Stats container
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown(f'<span class="metric-badge">⏱️ API Latency: {api_latency:.2f}s</span>', unsafe_allow_html=True)
            st.markdown(f'<span class="metric-badge-alt">⚡ Total RAG Latency: {total_latency:.2f}s</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 3. Sources Panel (expander)
            st.markdown("### 📂 Source Material Context")
            with st.expander("Show Retrieved Documentation Chunks", expanded=False):
                if not context_chunks:
                    st.write("No document context was retrieved.")
                else:
                    st.write("The model evaluated these top 3 text chunks from the vector database:")
                    for idx, doc in enumerate(context_chunks, 1):
                        source = doc.metadata.get("source", "unknown")
                        chunk_idx = doc.metadata.get("chunk_index", "N/A")
                        
                        st.markdown(f"""
                        <div class="source-header">
                            <span>Chunk #{idx}</span>
                            <span>Source: {source} (Index {chunk_idx})</span>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f'<div class="source-block">{doc.page_content}</div>', unsafe_allow_html=True)

    # Sidebar details for Upwork Mode
    with st.sidebar:
        st.markdown("### 📋 Quick Evaluation Queries")
        st.write("Click or copy sample queries to test:")
        st.info("**Question A (Rate Limits)**\n\nWhat is the specific request-per-second rate limit for the Upwork API, and is it enforced per Key or per IP?")
        st.info("**Question B (OAuth Validity)**\n\nHow long is an OAuth access token valid for?")
        st.info("**Question C (Client Credentials)**\n\nCan I use a Client Credentials Grant to access a user's private contract details?")
        st.info("**Question D (Shopify Guard)**\n\nHow do I configure a Shopify webhook?")

else:
    # ------------------ EXPERIMENTAL PHYSICS SANDBOX MODE ------------------
    st.markdown('<div class="subtitle">🧪 Experimental Physics Sandbox</div>', unsafe_allow_html=True)
    
    # Render page from experimental_modes.py
    render_physics_sandbox_page(st)
    
    # Sidebar details for Physics Mode
    with st.sidebar:
        st.markdown("### 📋 Speculative Physics Queries")
        st.write("Test the reality-grounded behavioral guard with these sample queries:")
        st.info("**Query 1 (Anti-gravity)**\n\nWhat is the theoretical particle interaction required for localized spacetime manipulation?")
        st.info("**Query 2 (Warp Metrics)**\n\nHow can we design an Alcubierre warp drive using negative energy fields?")
        st.info("**Query 3 (Standard Physics)**\n\nWhat is the current scientific consensus on gravitational waves?")
