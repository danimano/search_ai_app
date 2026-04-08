from llama_cpp import Llama
import requests
import re
import os

llm = Llama(model_path="./model.gguf", n_ctx=4096, verbose=False)


def load_instruction():
    if os.path.exists("pass1_instruction.txt"):
        with open("pass1_instruction.txt", 'r') as f: return f.read().strip()
    return "Output exactly 3 different search queries on separate lines based on the user's prompt."


def search_wikipedia_multi(queries: list) -> str:
    headers = {'User-Agent': 'MyLocalAISearch/1.0 (contact@example.com)'}
    all_context = ""
    seen_titles = set()

    for query in queries:
        clean_query = re.sub(r'^\d+\.\s*', '', query).strip(' "\'')
        if not clean_query: continue

        search_url = "https://en.wikipedia.org/w/api.php"
        params = {"action": "query", "list": "search", "srsearch": clean_query, "format": "json", "srlimit": 1}

        try:
            r = requests.get(search_url, params=params, headers=headers, timeout=3)
            results = r.json().get('query', {}).get('search', [])
            if results:
                title = results[0]['title']
                if title not in seen_titles:
                    seen_titles.add(title)
                    sum_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
                    sum_r = requests.get(sum_url, headers=headers, timeout=3)
                    if sum_r.status_code == 200:
                        all_context += f"--- SOURCE: {title} ---\n{sum_r.json().get('extract', '')}\n\n"
        except Exception:
            continue
    return all_context.strip() if all_context else "No context found."


def pass_1_extract_queries(user_prompt: str) -> list:
    instruction = load_instruction()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=50, temperature=0.2
    )
    raw_output = response["choices"][0]["message"]["content"]
    return [q.strip(' "-.\'') for q in raw_output.split('\n') if q.strip()][:3]


def pass_2_generate_answer(user_prompt: str, context: str) -> str:
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Use ONLY the provided sources to answer."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_prompt}"}
        ],
        max_tokens=300, temperature=0.1
    )
    return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    question = input("Enter your question: ")
    print("[AI] Thinking...")
    queries = pass_1_extract_queries(question)
    print(f"[App] Extracted Queries: {queries}")
    context = search_wikipedia_multi(queries)
    answer = pass_2_generate_answer(question, context)
    print(f"\n--- Final Answer ---\n{answer}\n--------------------")