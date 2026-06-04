import os
import sys
import logging
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging with a clean console format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Ensure stdout uses UTF-8 to prevent encoding errors on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Splits the input text into chunks using RecursiveCharacterTextSplitter.
    """
    logger.info(f"Chunking document (size={chunk_size}, overlap={chunk_overlap})...")
    
    # RecursiveCharacterTextSplitter splits by paragraphs, sentences, and words to avoid truncating content
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    chunks = splitter.split_text(text)
    logger.info(f"Created {len(chunks)} chunks from source text.")
    return chunks

def find_exact_overlap(chunk_a: str, chunk_b: str) -> str:
    """
    Finds the exact overlapping substring where the end of chunk_a matches the start of chunk_b.
    """
    max_len = min(len(chunk_a), len(chunk_b))
    for i in range(max_len, 0, -1):
        if chunk_a[-i:] == chunk_b[:i]:
            return chunk_a[-i:]
    return ""

def test_and_debug_chunking():
    # Load env and read document
    load_dotenv()
    document_path = os.getenv("DOCUMENT_PATH", "upwork_api_reference.txt")
    
    if not os.path.exists(document_path):
        logger.error(f"Error: Source file '{document_path}' not found. Run ingest.py first.")
        return
        
    with open(document_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    # Split text
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    
    if len(chunks) < 2:
        logger.warning("Not enough chunks created to compare overlap.")
        return
        
    chunk_0 = chunks[0]
    chunk_1 = chunks[1]
    
    # Find overlapping segment
    overlap = find_exact_overlap(chunk_0, chunk_1)
    overlap_len = len(overlap)
    
    # ANSI escape codes for styling
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    
    print("\n" + "="*80)
    print(" CHUNKING AND OVERLAP DEBUGGER ".center(80, "#"))
    print("="*80)
    print(f"Total Chunks Created: {len(chunks)}")
    print(f"Identified Overlap Length: {overlap_len} characters")
    print("="*80)
    
    # 1. Print Chunk 0 with overlap highlighted at the end
    print(f"{CYAN}[CHUNK #0 - Total Length: {len(chunk_0)} characters]{RESET}")
    print("-" * 80)
    if overlap:
        non_overlap_a = chunk_0[:-overlap_len]
        print(non_overlap_a, end="")
        print(f"{GREEN}<<< OVERLAP START >>>{overlap}{GREEN}<<< OVERLAP END >>>{RESET}")
    else:
        print(chunk_0)
    print("-" * 80)
    
    # 2. Print Chunk 1 with overlap highlighted at the beginning
    print(f"\n{CYAN}[CHUNK #1 - Total Length: {len(chunk_1)} characters]{RESET}")
    print("-" * 80)
    if overlap:
        non_overlap_b = chunk_1[overlap_len:]
        print(f"{GREEN}<<< OVERLAP START >>>{overlap}{GREEN}<<< OVERLAP END >>>{RESET}", end="")
        print(non_overlap_b)
    else:
        print(chunk_1)
    print("-" * 80)
    
    # 3. Print the highlighted standalone overlap segment
    print(f"\n{YELLOW}[VISUAL OVERLAP HIGHLIGHT]{RESET}")
    print("-" * 80)
    if overlap:
        # Show exact overlap representation
        print(f"The overlapping text block between Chunk #0 and Chunk #1 is:")
        print(f"{GREEN}------------------------------------------------------------\n{overlap}\n------------------------------------------------------------{RESET}")
    else:
        print("No exact text overlap found between Chunk #0 and Chunk #1.")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_and_debug_chunking()
