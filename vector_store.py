import os
import sys
import logging
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Import chunking logic from chunking.py
from chunking import chunk_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Ensure stdout uses UTF-8 to prevent console character encoding errors
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

def initialize_embeddings() -> HuggingFaceEmbeddings:
    """
    Initializes the sentence-transformers/all-MiniLM-L6-v2 embeddings model.
    """
    logger.info("Initializing HuggingFaceEmbeddings (sentence-transformers/all-MiniLM-L6-v2)...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        logger.info("Embeddings model initialized successfully.")
        return embeddings
    except Exception as e:
        logger.critical(f"Failed to initialize embeddings: {e}")
        raise e

def create_vector_store(text: str, persist_directory: str = "./chroma_db") -> Chroma:
    """
    Chunks the input text, generates embeddings, and saves them to a local Chroma vector database.
    """
    # 1. Chunk the document
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    
    # Convert string chunks to LangChain Document objects
    documents = [Document(page_content=chunk, metadata={"source": "upwork_api_reference.txt", "chunk_index": idx}) 
                 for idx, chunk in enumerate(chunks)]
    
    # 2. Get embeddings model
    embeddings = initialize_embeddings()
    
    # 3. Create and persist Chroma DB
    logger.info(f"Generating vectors and persisting database to '{persist_directory}'...")
    try:
        # Chroma automatically persists the data to the directory in recent versions
        db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        logger.info("Vector database creation and persistence completed successfully.")
        return db
    except Exception as e:
        logger.critical(f"Failed to create vector store: {e}")
        raise e

def query_vector_store(query: str, persist_directory: str = "./chroma_db") -> tuple:
    """
    Loads the persistent Chroma vector store and queries it using similarity search.
    """
    embeddings = initialize_embeddings()
    logger.info(f"Loading persistent database from '{persist_directory}'...")
    
    try:
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        logger.info(f"Performing similarity search for query: '{query}'...")
        results = db.similarity_search(query, k=1)
        
        if not results:
            logger.warning("No documents retrieved.")
            return None, 0.0
            
        # Perform similarity search with score
        results_with_scores = db.similarity_search_with_score(query, k=1)
        score = results_with_scores[0][1] if results_with_scores else 0.0
        
        return results[0], score
    except Exception as e:
        logger.error(f"Error querying vector store: {e}")
        raise e

def test_and_debug_vector_store():
    load_dotenv()
    
    document_path = os.getenv("DOCUMENT_PATH", "upwork_api_reference.txt")
    persist_db_dir = "./chroma_db"
    
    # Ingest document
    if not os.path.exists(document_path):
        logger.error(f"Source reference document '{document_path}' not found. Run ingest.py first.")
        return
        
    with open(document_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    # Create the vector store
    db = create_vector_store(text, persist_directory=persist_db_dir)
    
    # Verify that the directory was physically created
    print("\n" + "="*80)
    print(" VECTOR STORE PHYSICAL DIRECTORY VERIFICATION ".center(80, "#"))
    print("="*80)
    if os.path.exists(persist_db_dir) and os.listdir(persist_db_dir):
        print(f"Physical Database Directory: DETECTED (Path: {os.path.abspath(persist_db_dir)})")
        print("Database Directory Contents:")
        for item in os.listdir(persist_db_dir):
            item_path = os.path.join(persist_db_dir, item)
            size_kb = os.path.getsize(item_path) / 1024 if os.path.isfile(item_path) else 0
            file_type = "File" if os.path.isfile(item_path) else "Directory"
            print(f"  - {item:<30} ({file_type:<9} - {size_kb:.2f} KB)")
    else:
        print(f"Physical Database Directory: NOT DETECTED or EMPTY at '{persist_db_dir}'")
    print("="*80 + "\n")
    
    # Perform similarity search query
    test_query = "OAuth token validity"
    closest_doc, score = query_vector_store(test_query, persist_directory=persist_db_dir)
    
    # Styling escape codes
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    RESET = '\033[0m'
    
    print("\n" + "="*80)
    print(" SIMILARITY SEARCH VERIFICATION RESULT ".center(80, "#"))
    print("="*80)
    print(f"Query: '{test_query}'")
    if closest_doc:
        print(f"Match Similarity Distance Score: {score:.4f}")
        print(f"Source Metadata: {closest_doc.metadata}")
        print(f"\n{CYAN}[CLOSET RETRIEVED DOCUMENT TEXT]{RESET}")
        print("-" * 80)
        # Safely print text using console-safe encoding
        safe_text = closest_doc.page_content.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
        print(safe_text)
        print("-" * 80)
    else:
        print("No match found.")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_and_debug_vector_store()
