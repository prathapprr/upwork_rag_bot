import os
import sys
import logging
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from vector_store import initialize_embeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Ensure stdout uses UTF-8 to prevent character encoding crashes on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

def get_relevant_context(query: str, persist_directory: str = "./chroma_db") -> list:
    """
    Loads the persistent Chroma vector store and retrieves the top 3 (k=3)
    most relevant document chunks for the given query.
    """
    logger.info("Initializing similarity retrieval...")
    
    # 1. Initialize local embeddings function
    embeddings = initialize_embeddings()
    
    # 2. Check if the database exists
    if not os.path.exists(persist_directory):
        logger.error(f"Database directory '{persist_directory}' not found. Run vector_store.py first.")
        raise FileNotFoundError(f"Database at '{persist_directory}' does not exist.")
        
    # 3. Load Chroma vector database
    logger.info(f"Connecting to database at '{persist_directory}'...")
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    # 4. Query vector store for top 3 documents
    logger.info(f"Retrieving top 3 chunks for query: '{query}'...")
    try:
        retrieved_docs = db.similarity_search(query, k=3)
        logger.info(f"Successfully retrieved {len(retrieved_docs)} chunks.")
        return retrieved_docs
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise e

def test_semantic_retrieval():
    load_dotenv()
    persist_db_dir = "./chroma_db"
    
    # The three assignment evaluation questions
    questions = [
        "What is the specific request-per-second rate limit for the Upwork API, and is it enforced per Key or per IP?",
        "How long is an OAuth access token valid for?",
        "Can I use a Client Credentials Grant to access a user's private contract details?"
    ]
    
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    
    print("\n" + "="*80)
    print(" RAG PIPELINE - PART B1: SEMANTIC RETRIEVAL TESTER ".center(80, "#"))
    print("="*80)
    
    for idx, q in enumerate(questions):
        print(f"\n{YELLOW}[QUESTION {chr(65+idx)}] {q}{RESET}")
        print("="*80)
        
        try:
            chunks = get_relevant_context(q, persist_directory=persist_db_dir)
            
            for rank, doc in enumerate(chunks, 1):
                chunk_len = len(doc.page_content)
                source_file = doc.metadata.get("source", "unknown")
                chunk_idx = doc.metadata.get("chunk_index", "N/A")
                
                print(f"\n{GREEN}Rank #{rank} | Length: {chunk_len} chars | Source: {source_file} (Chunk {chunk_idx}){RESET}")
                print("-" * 80)
                
                # Safely encode text using console-safe encoding
                safe_text = doc.page_content.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
                print(safe_text.strip())
                print("-" * 80)
                
        except Exception as e:
            print(f"Error executing retrieval for Question {chr(65+idx)}: {e}")
            
    print("\n" + "="*80)
    print(" RETRIEVAL TESTING COMPLETE ".center(80, "#"))
    print("="*80 + "\n")

if __name__ == "__main__":
    test_semantic_retrieval()
