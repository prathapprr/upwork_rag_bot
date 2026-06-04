import os
import sys
import time
import logging
import requests
from dotenv import load_dotenv

# Import retrieval function from retrieval.py
from retrieval import get_relevant_context

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

def generate_rag_response(query: str, context_chunks: list) -> tuple:
    """
    Constructs the prompt, calls the DeepInfra API using a direct requests call,
    and returns the model's text response and the response latency in seconds.
    """
    load_dotenv()
    
    # 1. Retrieve configuration values safely from environment variables
    api_key = os.environ.get("DEEPINFRA_API_KEY")
    api_base = os.environ.get("DEEPINFRA_API_BASE", "https://api.deepinfra.com/v1/openai")
    model = os.environ.get("DEEPINFRA_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
    
    if not api_key:
        logger.error("DEEPINFRA_API_KEY environment variable is missing.")
        raise ValueError("DEEPINFRA_API_KEY environment variable must be set in your .env file.")

    # 2. Build the contiguous context text from retrieved chunks
    context_text = ""
    if context_chunks:
        context_text = "\n\n".join([
            f"--- Context Block #{idx+1} (Source: {doc.metadata.get('source')}, Chunk: {doc.metadata.get('chunk_index')}) ---\n{doc.page_content}"
            for idx, doc in enumerate(context_chunks)
        ])
    else:
        context_text = "NO CONTEXT BLOCKS PROVIDED."

    # 3. Construct System Prompt & Hallucination Guard
    system_prompt = (
        "You are a Senior Upwork API Consultant. Answer user queries strictly using the provided context blocks.\n"
        "CRITICAL: If the answer to the user's question cannot be completely derived from the provided context blocks, "
        "you must reply exactly with: 'I'm sorry, but the provided documentation does not contain that information.' "
        "Do not attempt to use outside knowledge or hallucinate details. Do not explain your inability to answer beyond the exact sentence."
    )

    user_content = (
        f"Provided Context Blocks:\n{context_text}\n\n"
        f"User Question: {query}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    # 4. Invoke the DeepInfra API and measure latency
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0  # Force maximum determinism and accuracy
    }

    logger.info(f"Sending request to DeepInfra API endpoint for query: '{query}'...")
    start_time = time.time()
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        latency = time.time() - start_time
        
        # Check HTTP response status
        if response.status_code != 200:
            logger.error(f"API Request Failed: Status Code {response.status_code} - {response.text}")
            return f"Error: DeepInfra API returned status code {response.status_code}.", latency
            
        response_data = response.json()
        answer = response_data["choices"][0]["message"]["content"]
        logger.info(f"Received API response in {latency:.2f} seconds.")
        return answer.strip(), latency
        
    except requests.exceptions.Timeout:
        latency = time.time() - start_time
        logger.error("DeepInfra API request timed out.")
        return "Error: DeepInfra API request timed out.", latency
    except Exception as e:
        latency = time.time() - start_time
        logger.error(f"Failed to communicate with DeepInfra API: {e}")
        return f"Error: Request failed due to: {e}", latency

def run_complete_rag_pipeline(query: str) -> tuple:
    """
    Executes the entire RAG pipeline:
    1. Retrieval of top 3 chunks.
    2. LLM response generation with Hallucination Guard.
    """
    try:
        chunks = get_relevant_context(query)
    except Exception as e:
        logger.error(f"Retrieval phase failed: {e}")
        chunks = []
        
    answer, latency = generate_rag_response(query, chunks)
    return answer, chunks, latency

def run_rag_evaluations():
    # Load environment variables
    load_dotenv()
    
    # 3 core evaluation questions + 1 fake question to test Hallucination Guard
    test_queries = [
        # Question A (Expect Hallucination Guard to trigger due to missing context)
        "What is the specific request-per-second rate limit for the Upwork API, and is it enforced per Key or per IP?",
        
        # Question B (Expect successful retrieval and answer)
        "How long is an OAuth access token valid for?",
        
        # Question C (Expect successful retrieval and answer based on client context)
        "Can I use a Client Credentials Grant to access a user's private contract details?",
        
        # Fake Question (Expect Hallucination Guard to trigger)
        "How do I configure a Shopify webhook?"
    ]
    
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    
    print("\n" + "="*80)
    print(" RAG PIPELINE - PART B2: API INTEGRATION & PROMPTING TEST ".center(80, "#"))
    print("="*80)
    
    for idx, q in enumerate(test_queries, 1):
        label = f"Query #{idx}"
        if idx == 4:
            label = "Hallucination Guard Test (Shopify)"
            
        print(f"\n{YELLOW}[{label}] {q}{RESET}")
        print("="*80)
        
        # Run pipeline
        answer, chunks, latency = run_complete_rag_pipeline(q)
        
        print(f"Latency: {latency:.4f} seconds")
        print(f"Retrieved Chunks: {len(chunks)}")
        print(f"\n{CYAN}[GENERATED RAG ANSWER]{RESET}")
        print("-" * 80)
        # Safe print encoding
        safe_answer = answer.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
        print(safe_answer)
        print("-" * 80)
        
    print("\n" + "="*80)
    print(" RAG INTEGRATION TESTING COMPLETE ".center(80, "#"))
    print("="*80 + "\n")

if __name__ == "__main__":
    run_rag_evaluations()
