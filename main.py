import os
from datetime import datetime
import requests
import re
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
REPO_ID = "unsloth/Qwen3.5-2B-GGUF"
FILENAME = "Qwen3.5-2B-Q4_K_M.gguf"

print("1. Checking/Downloading Model from HuggingFace...")
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

print("2. Loading Model into RAM (Llama.cpp)...")
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    verbose=False
)

def load_instruction(filename):
    """Helper to read the instruction text file."""
    if not os.path.exists(filename):
        return "Extract search keywords for Wikipedia. Output only keywords."
    with open(filename, 'r') as f:
        return f.read().strip()

# ---------------------------------------------------------
# Core Functions
# ---------------------------------------------------------

def pass_1_extract_query(user_prompt: str) -> list:
    """Pass 1: Asks the LLM to convert the user's natural language into a search query."""
    instruction = load_instruction("search_prompt.txt")

    system_prompt = f"{instruction}\n\nToday is {datetime.now().strftime('%A, %B %d, %Y')}."

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=50,
        temperature=0.1
    )
    query = response["choices"][0]["message"]["content"].strip(' "\'')
    queries = [q.strip(' "-.\'') for q in query.split('\n') if q.strip()]
    return queries[:3]

def search_wikipedia_multi(queries: list) -> str:
    headers = {'User-Agent': 'MyLocalAISearch/1.0 (contact@example.com)'}
    all_context = ""
    seen_titles = set()

    for query in queries:
        clean_query = re.sub(r'^\d+\.\s*', '', query).strip(' "\'')
        if not clean_query: continue

        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query", "list": "search", "srsearch": clean_query,
            "format": "json", "srlimit": 3
        }

        try:
            r = requests.get(search_url, params=params, headers=headers, timeout=3)
            results = r.json().get('query', {}).get('search', [])

            for result in results:
                title = result['title']
                raw_snippet = result.get('snippet', '')
                clean_snippet = re.sub(r'<[^>]+>', '', raw_snippet)

                if title not in seen_titles:
                    seen_titles.add(title)
                    sum_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
                    sum_r = requests.get(sum_url, headers=headers, timeout=3)

                    if sum_r.status_code == 200:
                        summary = sum_r.json().get('extract', '')
                        wiki_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

                        all_context += (
                            f"--- SOURCE {len(seen_titles)}: {title} ---\n"
                            f"URL: {wiki_url}\n"
                            f"Summary: {summary}\n"
                            f"Keyword Match: {clean_snippet}...\n\n"
                        )
        except Exception:
            continue

    return all_context.strip() if all_context else "Error: Could not find relevant Wikipedia pages."

def pass_2_generate_answer(user_prompt: str, context: str) -> str:
    """Pass 2: Feeds the Wiki snippets to the LLM to answer the user."""
    today = datetime.now().strftime("%A, %B %d, %Y")

    system_prompt = (
        f"Today is {today}. You are a factual assistant. "
        "Use ONLY the provided Wikipedia entries to answer in 150 words or less. "
        "If the entries contradict your internal memory, TRUST THE WIKIPEDIA. "
        "Provide citations to sources as [1], [2], etc. "
        "and make references to the actual wikipedia articles at the end of your answer."
    )

    combined_user_content = f"CONTEXT FROM WIKIPEDIA:\n{context}\n\nQUESTION: {user_prompt}"

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": combined_user_content}
        ],
        max_tokens=512,
        temperature=0.3,
        stream=True
    )

    print("\n--- Final Answer ---")
    final_text = ""
    for chunk in response:
        if "content" in chunk["choices"][0]["delta"]:
            token = chunk["choices"][0]["delta"]["content"]
            print(token, end="", flush=True)
            final_text += token
    print("\n--------------------\n")
    return final_text


# ---------------------------------------------------------
# Execution Flow
# ---------------------------------------------------------

if __name__ == "__main__":
    user_input = input("\nEnter your question: ")

    print(f"\n[AI] Thinking...")
    search_query = pass_1_extract_query(user_input)
    print(f"[App] Extracted Search Queries: -> '{search_query}'")

    print(f"[App] Fetching data from Wikipedia...")
    wiki_context = search_wikipedia_multi(search_query)
    print(f"[App] Context length passed to LLM: {len(wiki_context)} characters")

    pass_2_generate_answer(user_input, wiki_context)