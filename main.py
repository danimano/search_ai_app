from llama_cpp import Llama
import requests

# Initialize the 2B model
llm = Llama(model_path="./model.gguf", n_ctx=4096, verbose=False)

def search_wikipedia(query: str) -> str:
    # Naive search without cleaning or headers
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query", "list": "search",
        "srsearch": query, "format": "json", "srlimit": 3
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        compiled_results = ""
        for item in data.get('query', {}).get('search', []):
            title = item['title']
            snippet = item.get('snippet', '').replace('<span class="searchmatch">', '').replace('</span>', '')
            compiled_results += f"Source ({title}): {snippet}...\n\n"
        return compiled_results.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def pass_1_extract_query(user_prompt: str) -> str:
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "Extract search keywords for Wikipedia. Output only keywords."},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=20, temperature=0.1
    )
    return response["choices"][0]["message"]["content"]

def pass_2_generate_answer(user_prompt: str, context: str) -> str:
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer the user's question using ONLY the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_prompt}"}
        ],
        max_tokens=200, temperature=0.1
    )
    return response["choices"][0]["message"]["content"]

if __name__ == "__main__":
    question = input("Enter your question: ")
    print("[AI] Thinking...")
    query = pass_1_extract_query(question)
    print(f"[App] Extracted Query: {query}")
    context = search_wikipedia(query)
    answer = pass_2_generate_answer(question, context)
    print(f"\n--- Final Answer ---\n{answer}\n--------------------")