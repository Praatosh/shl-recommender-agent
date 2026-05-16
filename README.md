# SHL Assessment Recommender Agent

An intelligent, conversational AI agent designed to help recruiters and hiring managers discover the most relevant SHL assessments, simulations, and development tools for their hiring needs.

## Overview
The SHL Recommender Agent utilizes a **Retrieval-Augmented Generation (RAG)** architecture. It embeds the entire SHL product catalog using Sentence Transformers and stores it in a local FAISS vector database. When a user asks a question, the system retrieves the most relevant products and uses a Groq-hosted Large Language Model (`llama-3.3-70b-versatile`) to generate a conversational, context-aware recommendation without hallucinations.

## Features
* **Stateless Chat Interface**: Handles full conversation history to allow follow-up questions and refinements.
* **Vector Search**: Uses `sentence-transformers/all-MiniLM-L6-v2` and `faiss-cpu` for lightning-fast semantic retrieval of catalog items.
* **Guardrails**: Prompt engineering prevents the LLM from inventing fake assessments or hallucinating URLs.
* **Docker Ready**: Fully containerized and ready for cloud deployment.

## Tech Stack
* **Framework**: FastAPI, Uvicorn
* **LLM Provider**: Groq API
* **Embeddings**: Sentence-Transformers
* **Vector Store**: FAISS
* **Deployment**: Docker, Render

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

6. **Test the API**
   Open your browser and navigate to `http://127.0.0.1:8000/docs` to use the interactive Swagger UI.

## Usage & API Reference

The application exposes a single functional POST endpoint for conversation at `/chat`.

### Sample Query
```bash
curl -X POST "https://shl-recommender-0kcr.onrender.com/chat" \
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
The API responds with a structured JSON object containing a conversational reply and a strict array of valid SHL products.
```json
{
  "reply": "For an entry-level call center position handling irate customers and data entry, I recommend the Customer Service Phone Simulation and Data Entry Alphanumeric Split Screen - US assessments...",
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
