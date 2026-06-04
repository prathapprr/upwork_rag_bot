import os
import sys
import time
import requests
import logging
from dotenv import load_dotenv

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

def generate_physics_response(query: str) -> tuple:
    """
    Calls the DeepInfra API with a custom system prompt that enforces reality-grounded
    behavior when users ask about speculative technology like anti-gravity.
    Returns (answer, latency).
    """
    load_dotenv()
    
    # 1. Configuration
    api_key = os.environ.get("DEEPINFRA_API_KEY")
    api_base = os.environ.get("DEEPINFRA_API_BASE", "https://api.deepinfra.com/v1/openai")
    model = os.environ.get("DEEPINFRA_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
    
    if not api_key:
        logger.error("DEEPINFRA_API_KEY environment variable is missing.")
        raise ValueError("DEEPINFRA_API_KEY environment variable must be set in your .env file.")

    # 2. Construct System Prompt & Speculative Physics Behavioral Guards
    system_prompt = (
        "You are a Reality-Grounded Physics Research Assistant specializing in theoretical and speculative physics.\n\n"
        "CRITICAL BEHAVIORAL GUARDS:\n"
        "1. If the user asks about speculative technology or science fiction premises (such as anti-gravity, localized spacetime manipulation, or warp drives), you must remain calm, neutral, and grounded in objective facts.\n"
        "2. Acknowledge the user's creative curiosity or theoretical interest with empathy, but gently distinguish these speculative premises from established engineering realities and physical laws without being argumentative or condescending.\n"
        "3. Safely pivot the user back to standard, verifiable concepts in physics (such as general relativity, electromagnetic fields, quantum field theory, gravitational wave detection, or standard propulsion models) and explain the current scientific consensus."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User Query: {query}"}
    ]

    # 3. Call DeepInfra API
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2  # Slightly higher than 0.0 to allow for conceptual explanations, but keeping it grounded
    }

    logger.info(f"Sending request to DeepInfra API for Physics Sandbox query: '{query}'...")
    start_time = time.time()
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        latency = time.time() - start_time
        
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

def render_physics_sandbox_page(st_context):
    """
    Renders the UI elements of the Antigravity Physics Reference Mode inside Streamlit.
    """
    st_context.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st_context.subheader("⚛️ Antigravity Physics Reference Mode")
    st_context.write("Explore theoretical and speculative physics in a reality-grounded sandbox environment. Ask questions about spacetime dynamics, energy fields, and gravity manipulation models.")
    
    # Text input form for physics query
    with st_context.form(key="physics_form"):
        physics_query = st_context.text_area(
            label="Enter your theoretical physics question:",
            placeholder="e.g., What is the theoretical particle interaction required for localized spacetime manipulation?",
            height=100
        )
        submit_physics = st_context.form_submit_button(label="Analyze Theoretical Physics")
        
    st_context.markdown('</div>', unsafe_allow_html=True)
    
    # Process physics query
    if submit_physics:
        if not physics_query.strip():
            st_context.warning("Please enter a valid theoretical physics query.")
        else:
            with st_context.spinner("Analyzing theory and formulating reality-grounded perspective..."):
                t_start = time.time()
                answer, api_latency = generate_physics_response(physics_query)
                t_stop = time.time()
                total_latency = t_stop - t_start
                
            st_context.success("Theoretical Analysis Complete!")
            
            # Display Answer Card
            st_context.markdown("### 🔍 Scientific Analysis")
            st_context.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)
            
            # Display Latency Metrics
            st_context.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st_context.markdown(f'<span class="metric-badge">⏱️ API Latency: {api_latency:.2f}s</span>', unsafe_allow_html=True)
            st_context.markdown(f'<span class="metric-badge-alt">⚡ Total Sandbox Latency: {total_latency:.2f}s</span>', unsafe_allow_html=True)
            st_context.markdown('</div>', unsafe_allow_html=True)
            
            # Display additional helpful resources or contextual guidelines
            st_context.info("**Sandbox Guideline**: Speculative technologies like localized gravity manipulation or warp metrics are analyzed based on general relativity solutions (such as Alcubierre geometry or Einstein-Maxwell couplings) while highlighting engineering limits like exotic matter requirements.")
