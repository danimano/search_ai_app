from llama_cpp import Llama
import requests
import re

llm = Llama(model_path="./model.gguf", n_ctx=4096, verbose=False)


def search_wikipedia(query: str) -> str:
    clean_query = " ".join(query.split()).strip(' "\'')
    headers = {'User-Agent': 'MyLocalAISearch/1.0 (contact@example.com)'}

    # STEP 1: Find the best matching page title
    search_url = "https://en.wikipedia.org/w/api.php"
    params = {"action": "query", "list": "search", "srsearch": clean_query, "format": "json", "srlimit": 1}

    try:
        r = requests.get(search_url, params=params, headers=headers, timeout=5)
        search_results = r.json().get('query', {}).get('search', [])

        if not search_results: return f"No results for '{clean_query}'"

        best_title = search_results[0]['title']

        # STEP 2: Upgrade to Summary API for high-density context
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{best_title.replace(' ', '_')}"
        summary_resp = requests.get(summary_url, headers=headers, timeout=5)

        if summary_resp.status_code == 200:
            summary_data = summary_resp.json()
            return f"--- Source: {best_title} ---\n{summary_data.get('extract', '')}"
        return "Error fetching summary."
    except Exception as e:
        return f"Error: {str(e)}"


def pass_1_extract_query(user_prompt: str) -> str:
    response = llm.create_chat_completion(
        messages=[
            {"role": "system",
             "content": "Extract search keywords for Wikipedia. Output only keywords. Do not answer the question."},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=20, temperature=0.1
    )
    return response["choices"][0]["message"]["content"]


def pass_2_generate_answer(user_prompt: str, context: str) -> str:
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are an expert researcher. Answer using ONLY the provided context."},
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