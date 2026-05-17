# SHL Conversational Assessment Recommender — Approach Document

## 1. Architecture Overview

The system uses a **stateless RAG (Retrieval-Augmented Generation)** architecture with a single-agent design. Every request carries the full conversation history, eliminating server-side session management.

```
User Request (full history)
       ↓
  FastAPI Endpoint (/chat)
       ↓
  Conversation Analyzer
  ├─ Fast Path: Regex intent detection (~0ms)
  └─ Slow Path: LLM analysis (~1-2s)
       ↓
  Intent Router
  ├─ greeting    → static response
  ├─ off_topic   → polite refusal
  ├─ confirm     → re-emit list + EOC=true
  ├─ compare     → retrieve + compare prompt
  ├─ refine      → retrieve + update list
  └─ recommend   → retrieve + recommend prompt
       ↓
  FAISS Vector Search (multi-query)
       ↓
  LLM Grounded Generation (JSON mode)
       ↓
  Post-Generation Validation
  ├─ URL existence check against catalog
  ├─ Name fuzzy-match fallback
  └─ Schema enforcement via Pydantic
       ↓
  Structured JSON Response
```

**Key design decisions:**
- **Two-tier analysis**: Regex fast-path handles ~40% of intents (greetings, confirmations, injections) instantly. Only complex queries hit the LLM analyzer.
- **Multi-query search**: The LLM decomposes "Java developer with leadership skills" into separate queries, improving recall by searching multiple facets.
- **Post-generation validation**: Every recommended URL is verified against the catalog. Hallucinated URLs are either fixed (via name lookup) or dropped.

## 2. Retrieval Strategy

**Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, fast inference, good semantic quality).

**Index**: FAISS `IndexFlatIP` with L2-normalized embeddings — inner product equals cosine similarity. For ~400 catalog items, exact search is fast enough (~1ms).

**Embedding text**: Each catalog item is embedded as a rich multi-field string combining name, description, categories, job levels, and duration. This allows a single embedding to capture multiple search dimensions.

**Search flow**:
1. LLM generates 2-4 search queries from conversation context
2. Each query searches FAISS independently
3. Results are merged with max-score deduplication
4. Top-K candidates (default 15) are passed to the LLM as grounding context

**Tradeoffs**:
- Dense retrieval alone may miss keyword matches (e.g., "OPQ32r" as an exact term). Mitigation: multi-query search with the LLM generating keyword-aware queries.
- Flat search is O(n) but acceptable at catalog size <1000. Would switch to IVF/HNSW at >10K items.

## 3. Prompt Strategy

Four specialized prompts, each optimized for its task:

| Prompt | Purpose | Key constraint |
|--------|---------|----------------|
| System | Agent persona + strict rules | Never hallucinate, only use catalog data |
| Analysis | Extract intent + requirements | Return structured JSON |
| Recommendation | Generate grounded response | Use EXACTLY the names/URLs from catalog |
| Comparison | Compare assessments | Only use retrieved catalog attributes |

**Guardrails embedded in prompts**:
- "NEVER invent assessment names or URLs"
- "ONLY recommend from CATALOG DATA provided below"
- Prompt injection defense: explicit instruction to deflect role-override attempts
- JSON mode enforced at the API level (Gemini: `response_mime_type="application/json"`, Groq: `response_format={"type": "json_object"}`)

## 4. Evaluation Methodology

**Schema correctness**: Automated test validates every response has `reply` (str), `recommendations` (list of {name, url, test_type}), `end_of_conversation` (bool).

**Recall@10**: Evaluated against 8 scenarios with expected assessments. Checks whether expected assessment names appear (substring match) in the recommendation list.

**Hallucination prevention**: Every recommended URL is checked against the scraped catalog. Any URL not in the catalog is flagged and either auto-corrected or dropped.

**Conversational quality**: Tested against 10 sample conversations (C1-C10) covering leadership hiring, full-stack engineering, call center staffing, and edge cases.

## 5. Tradeoffs & Decisions

| Decision | Chosen | Alternative | Why |
|----------|--------|-------------|-----|
| Embedding model | MiniLM-L6-v2 | e5-large-v2 | 5x faster, sufficient quality for <500 items |
| Vector index | FAISS flat | ChromaDB | No persistence overhead, exact search at this scale |
| LLM provider | Groq (Llama 3.3 70B) | Gemini Flash | Free tier, fast inference, JSON mode support |
| Architecture | Single agent | Multi-agent | Simpler, faster, fewer failure points |
| Session management | Stateless | Redis sessions | Assignment requirement; avoids infrastructure |
| Intent detection | Regex + LLM | LLM only | Saves 1-2s on trivial intents |

## 6. Known Limitations & Improvements

**Current limitations**:
- No BM25 hybrid search — pure dense retrieval may miss exact keyword matches
- No cross-encoder reranking — would improve precision at top positions
- No conversation summarization — long histories increase LLM token usage
- Single embedding model — no domain-fine-tuning on assessment terminology

**Planned improvements for Recall@10**:
1. Add BM25 keyword search alongside dense retrieval (hybrid fusion)
2. Use cross-encoder reranking (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`)
3. Metadata pre-filtering by job level and assessment category before ranking
4. Query expansion with synonym lists for common tech terms
5. Fine-tune embedding model on assessment description pairs

**Production improvements**:
- Add Redis caching for repeated queries
- Add rate limiting middleware
- Add structured logging with correlation IDs
- Add A/B testing framework for prompt variants
- Add monitoring dashboard for recall metrics
