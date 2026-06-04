import os
import sys
import logging
from dotenv import load_dotenv

# Configure logging with a clean console format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Ensure stdout uses UTF-8 to prevent charmap encoding errors on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    # Fallback if reconfigure is not supported (older python versions)
    pass

def run_sanity_check(file_path: str):
    """
    Performs the required sanity check on the ingested file.
    Calculates character count and prints a formatted sample slice.
    """
    logger.info("Initializing Data Ingestion Sanity Check...")
    
    # 1. Verify existence of target document
    if not os.path.exists(file_path):
        logger.error(f"Ingestion Failed: Reference file '{file_path}' not found.")
        print("\n" + "="*60)
        print("ERROR: TARGET FILE NOT FOUND")
        print(f"File path: {os.path.abspath(file_path)}")
        print("="*60)
        print("Troubleshooting Steps:")
        print("1. Ensure that you have run the PDF extraction process or")
        print("   manually created 'upwork_api_reference.txt' in the project root.")
        print("2. Check the DOCUMENT_PATH variable in your .env file.")
        print("="*60 + "\n")
        raise FileNotFoundError(f"The reference file '{file_path}' does not exist.")
    
    # 2. Ingest document content
    try:
        logger.info(f"Reading file: {file_path} ...")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logger.critical(f"Critical Ingestion Error: Unable to read file due to: {e}")
        raise e
    
    # 3. Calculate metrics
    char_count = len(content)
    line_count = len(content.splitlines())
    logger.info("Ingestion completed successfully.")
    
    # 4. Format and present sanity check results
    # Use standard ASCII characters for maximum compatibility across Windows terminals
    border_len = 80
    print("\n" + "+" + "="*(border_len-2) + "+")
    print("|" + " DATA INGESTION SANITY CHECK ".center(border_len-2, "#") + "|")
    print("+" + "="*(border_len-2) + "+")
    print(f"| Target Document:   {file_path:<57} |")
    print(f"| Total Characters:  {char_count:<57} |")
    print(f"| Total Lines:       {line_count:<57} |")
    print("+" + "="*(border_len-2) + "+")
    print("|" + " SAMPLE CONTENT (FIRST 500 CHARACTERS) ".center(border_len-2, "-") + "|")
    print("+" + "="*(border_len-2) + "+")
    
    # Extract first 500 characters slice
    sample_slice = content[:500]
    
    # Wrap sample slice for clean box presentation
    lines_to_print = sample_slice.split("\n")
    for line in lines_to_print:
        # Wrap long lines if necessary to fit nicely in 80-char terminal borders (max length of chunk = 74)
        chunks = [line[i:i+74] for i in range(0, len(line), 74)]
        if not chunks:
            print("|  " + " "*74 + "  |")
        for chunk in chunks:
            # Safely encode chunk to console encoding or replace unmappable chars
            safe_chunk = chunk.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
            print(f"|  {safe_chunk:<74}  |")
            
    print("+" + "="*(border_len-2) + "+")
    print("|" + " END OF SAMPLE SLICE ".center(border_len-2, "-") + "|")
    print("+" + "="*(border_len-2) + "+\n")
    
    return content

def main():
    # Load environment variables safely
    load_dotenv()
    
    # Retrieve the document path from environment variables, defaulting to 'upwork_api_reference.txt'
    document_path = os.getenv("DOCUMENT_PATH", "upwork_api_reference.txt")
    
    try:
        run_sanity_check(document_path)
    except FileNotFoundError:
        sys.exit(1)
    except Exception as e:
        logger.error(f"Execution terminated with error: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
