# SHL Assessment Recommender Agent

An intelligent, conversational AI agent designed to help recruiters and hiring managers discover the most relevant SHL assessments, simulations, and development tools for their hiring needs.

## Overview
The SHL Recommender Agent utilizes a **Retrieval-Augmented Generation (RAG)** architecture. It embeds the entire SHL product catalog using Sentence Transformers and stores it in a local FAISS vector database. When a user asks a question, the system retrieves the most relevant products and uses a Groq-hosted Large Language Model (`llama-3.3-70b-versatile`) to generate a conversational, context-aware recommendation without hallucinations.

## Architecture

```
User Request (full conversation history)
       ↓
  FastAPI Endpoint (/chat)
       ↓
  Conversation Analyzer (Regex fast-path + LLM deep analysis)
       ↓
  Intent Router → greeting | off_topic | confirm | compare | refine | recommend
       ↓
  FAISS Vector Search (multi-query, cosine similarity)
       ↓
  LLM Grounded Generation (JSON mode, catalog-only)
       ↓
  Post-Generation Validation (URL check, schema enforcement)
       ↓
  Structured JSON Response
```

## Features
* **Stateless Chat Interface**: Handles full conversation history to allow follow-up questions and refinements.
* **Two-Tier Intent Detection**: Regex fast-path for common intents (~0ms) + LLM for complex queries.
* **Vector Search**: Uses `sentence-transformers/all-MiniLM-L6-v2` and `faiss-cpu` for semantic retrieval.
* **Multi-Query Search**: LLM decomposes complex queries into multiple search facets for better recall.
* **Hallucination Prevention**: Post-generation URL validation ensures all recommendations exist in the catalog.
* **Guardrails**: Prompt injection defense, off-topic refusal, and scope adherence.
* **Async API**: Non-blocking endpoint using `asyncio.to_thread` for LLM/embedding operations.
* **Docker Ready**: Fully containerized and ready for cloud deployment.

## Tech Stack
* **Framework**: FastAPI, Uvicorn
* **LLM Provider**: Groq API (Llama 3.3 70B)
* **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
* **Vector Store**: FAISS (IndexFlatIP, cosine similarity)
* **Validation**: Pydantic v2
* **Deployment**: Docker, Render

## Project Structure

```
app/
├── main.py          # FastAPI app with lifespan, middleware, endpoints
├── config.py        # Centralized configuration via pydantic-settings
├── schemas.py       # Pydantic models for request/response validation
├── engine.py        # Recommendation engine (orchestrator)
├── analyzer.py      # Conversation analyzer (intent detection)
├── embeddings.py    # FAISS vector store & embedding generation
├── llm_client.py    # LLM abstraction layer (Gemini/Groq)
├── prompts.py       # All prompt templates (system, analysis, recommendation)
├── scraper.py       # Catalog loading & preprocessing
├── utils.py         # Shared utilities (TYPE_MAP, derive_test_type_code)
├── logger.py        # Centralized logging configuration
└── __init__.py
tests/
├── test_all.py      # Unit tests (schemas, utils, patterns)
└── test_evaluation.py  # Live API evaluation script
data/
├── catalog.json     # Pre-scraped SHL catalog (~400 assessments)
└── faiss_index/     # Persisted FAISS index files
```

## Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Praatosh/shl-recommender-agent.git
   cd shl-recommender-agent
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**
   Create a `.env` file based on the `.env.example`:
   ```env
   # LLM Provider
   LLM_PROVIDER=groq
   GROQ_API_KEY=your_api_key_here
   GROQ_MODEL=llama-3.3-70b-versatile
   
   # Embedding and Index configuration
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   FAISS_INDEX_PATH=data/faiss_index
   CATALOG_PATH=data/catalog.json
   ```

4. **Build the Vector Index**
   Before running the app for the first time, you must build the FAISS index from the catalog data:
   ```bash
   python build_index.py
   ```

5. **Run the Server**
   ```bash
   uvicorn app.main:app --reload
   ```

6. **Run Tests**
   ```bash
   python -m pytest tests/test_all.py -v
   ```

7. **Run Evaluation** (requires running server)
   ```bash
   python tests/test_evaluation.py
   ```

## Usage & API Reference

### Health Check
```bash
curl http://localhost:8000/health
```

### Chat Endpoint
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
           "messages": [
             {
               "role": "user",
               "content": "We need to evaluate candidates for an entry-level call center position. It involves handling irate customers and basic data entry. What should we use?"
             }
           ]
         }'
```

### Response Format
```json
{
  "reply": "For an entry-level call center position handling irate customers and data entry, I recommend...",
  "recommendations": [
    {
      "name": "Customer Service Phone Simulation",
      "url": "https://www.shl.com/products/product-catalog/view/customer-service-phone-simulation/",
      "test_type": "S"
    },
    {
      "name": "Data Entry Alphanumeric Split Screen - US",
      "url": "https://www.shl.com/products/product-catalog/view/data-entry-alphanumeric-split-screen-us/",
      "test_type": "S"
    }
  ],
  "end_of_conversation": false
}
```

## Deployment
This project is configured for seamless deployment on [Render](https://render.com/). 
The included `render.yaml` Blueprint automatically builds the FAISS index during the Docker build process and spins up the FastAPI web service.

Live API Documentation: [https://shl-recommender-0kcr.onrender.com/docs](https://shl-recommender-0kcr.onrender.com/docs)

## Approach Document
See [APPROACH.md](APPROACH.md) for a detailed explanation of:
- Architecture decisions
- Retrieval strategy
- Prompt engineering
- Evaluation methodology
- Tradeoffs and limitations
