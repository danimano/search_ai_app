# 🧠 WikiEdgeRAG: The Local Wikipedia-Exclusive RAG Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Llama.cpp](https://img.shields.io/badge/Llama.cpp-Local_Inference-orange.svg)
![Model](https://img.shields.io/badge/Model-Qwen_3.5_2B-blue.svg)
![Environment](https://img.shields.io/badge/Environment-Edge_Compute_(4GB_RAM)-lightgrey.svg)

## 📌 Overview
**WikiEdgeRAG** is a lightweight, locally-hosted Retrieval-Augmented Generation (RAG) pipeline optimized for highly constrained compute environments (e.g., 4GB RAM mobile/edge devices). 

**Note: This pipeline is purpose-built exclusively for Wikipedia.** It interfaces directly with the Wikipedia Search and REST APIs to provide fully grounded, cited, and hallucination-free answers using a local `Qwen3.5-2B` model via `llama.cpp`.

Rather than relying on heavy orchestration frameworks like LangChain, this pipeline is written close to the metal to minimize dependency bloat, strictly control token flow, and stream results in real-time.

## 🏗️ Architecture & Engineering Decisions

Building RAG pipelines with Small Language Models (SLMs) against Wikipedia's strict search engine introduces unique challenges. This project implements several defensive engineering patterns:

### 1. Automated Provisioning & Edge-Ready Weights
To make the pipeline truly plug-and-play, model weights are not hardcoded or manually managed. 
* **Solution:** Integrated `huggingface_hub` to automatically pull heavily quantized GGUF weights (`Qwen3.5-2B-Q4_K_M`) directly into RAM upon execution, ensuring the application stays highly portable and edge-device friendly.

### 2. Bypassing the "Wikipedia Search Ceiling" with Multi-Query Extraction
Wikipedia's standard search API is highly literal and often fails on historical, categorical, or vague queries.
* **Solution:** Implemented **Multi-Query Expansion**. The Pass 1 LLM extraction phase forces the model to generate up to three distinct search vectors. The Python layer dynamically queries all three through the Wikipedia API, casting a wider net and deduplicating results via a `seen_titles` hash set.

### 3. Context Window Optimization (`n_ctx=2048`)
To prevent Out-Of-Memory (OOM) crashes on 4GB RAM devices, context injection must be surgical.
* **Solution:** Instead of dumping full Wikipedia articles, the pipeline hits the Wikipedia API twice per query: first grabbing the specific **Search Snippet** (the exact keyword match), and then the **REST API Summary** (the broad context). This maximizes information density while comfortably fitting within a strict 2048 token context window limit.

### 4. Real-Time Streaming & Strict Grounding
High Time-To-First-Token (TTFT) ruins edge AI user experiences. Furthermore, SLMs are prone to hallucinating facts from their internal training data.
* **Solution:** Enabled `stream=True` in `llama.cpp` for a responsive, typewriter-style UI. The system prompt aggressively forces the model to "TRUST THE WIKIPEDIA" over its internal memory, enforcing inline citations (e.g., `[1]`) and appending Wikipedia URLs directly in the streamed output.

## 🚀 Tech Stack
* **Inference Engine:** `llama.cpp` (Local CPU/Edge optimized)
* **Language Model:** Qwen 3.5 2B (Quantized GGUF)
* **Retrieval APIs:** Wikipedia Search API & Wikipedia REST API
* **Cloud Hub:** HuggingFace Hub (for dynamic model downloading)

## 💻 Running Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/WikiEdgeRAG.git](https://github.com/YourUsername/WikiEdgeRAG.git)
   cd WikiEdgeRAG
   ```
2. **Install dependencies:**
   ```bash
   pip install llama-cpp-python requests huggingface_hub
   ```
3. **Configure Prompt (Optional):**
   Ensure you have a `search_prompt.txt` in the root directory to guide the Pass 1 search generation.
4. **Run the pipeline:**
   *The script will automatically download the Qwen model on first run.*
   ```bash
   python main.py
   ```

## 🛡️ Example Output
```text
Enter your question: who was the last king of hungary?

[AI] Thinking...
[App] Filtered Keywords: 'last king hungary'

--- Final Answer ---
SOURCE ANALYSIS:
- Source 1 (Kingdom of Hungary): Relevant, outlines the timeline of the monarchy.
- Source 2 (List of Hungarian monarchs): Relevant, establishes the end date of the kingdom (1918).
- Source 3 (Charles IV of Hungary): Highly relevant, explicitly names the final ruling monarch.

FINAL ANSWER:
The last King of Hungary was Charles IV, who reigned until the dissolution of the monarchy in 1918.

CITATIONS:
- [https://en.wikipedia.org/wiki/Kingdom_of_Hungary](https://en.wikipedia.org/wiki/Kingdom_of_Hungary)
- [https://en.wikipedia.org/wiki/List_of_Hungarian_monarchs](https://en.wikipedia.org/wiki/List_of_Hungarian_monarchs)
- [https://en.wikipedia.org/wiki/Charles_IV_of_Hungary](https://en.wikipedia.org/wiki/Charles_IV_of_Hungary)
```