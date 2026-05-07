# Lazy Loading Fix for Render Deployment

## Problem
The application was timing out on Render deployment because routers were initializing heavy RAG components (embedding models, vector stores) at **import time**, which took several minutes and prevented the app from binding to the port within Render's 10-minute timeout.

## Root Cause
In `ai_chat.py` and `hr_assistant.py`, RAG components were initialized at module level:
```python
# OLD - Blocks import
embedding_service = EmbeddingService(model_name=settings.EMBEDDING_MODEL)
vector_store = VectorStoreManager(settings.DATABASE_URL)
rag_retriever = RAGRetriever(vector_store, embedding_service)
```

This meant:
1. FastAPI imports routers during startup
2. Router imports trigger embedding model download (sentence-transformers)
3. Model download takes 3-5 minutes
4. App never reaches port binding
5. Render times out after 10 minutes

## Solution
Implemented **lazy loading** pattern - RAG components only initialize on first actual use:

### Changes Made

#### 1. `myapp/routers/ai_chat.py`
- Replaced module-level initialization with `get_rag_retriever()` async function
- Uses `asyncio.Lock()` to prevent race conditions
- Caches initialized instance for reuse
- Returns `None` if initialization fails (graceful degradation)

#### 2. `myapp/routers/hr_assistant.py`
- Same lazy loading pattern as ai_chat.py
- Independent initialization (separate cache)

#### 3. `myapp/services/ai.py`
- Added missing `logging` import
- Fixed logger usage in AIManager

#### 4. `myapp/startup.py`
- Disabled RAG data check on startup (was connecting to database)
- RAG now initializes completely on-demand

## Benefits
1. **Fast Startup**: App binds to port in ~10-20 seconds
2. **Graceful Degradation**: If RAG fails, app uses fallback mode
3. **On-Demand Loading**: Heavy models only load when actually needed
4. **Thread-Safe**: Lock prevents multiple simultaneous initializations
5. **Cached**: Once initialized, reused for all subsequent requests

## Deployment Flow
1. App starts → imports routers (fast, no heavy operations)
2. App binds to port → Render marks as healthy
3. First chat request → triggers RAG initialization (3-5 minutes)
4. Subsequent requests → use cached RAG instance (fast)

## Testing
```bash
# Syntax check
python -m py_compile myapp/routers/ai_chat.py myapp/routers/hr_assistant.py myapp/services/ai.py myapp/startup.py

# Local test
uvicorn main:app --host 0.0.0.0 --port 8000
# Should start in <30 seconds

# Test RAG lazy loading
curl -X POST http://localhost:8000/api/ai_chat/request \
  -H "Content-Type: application/json" \
  -d '{"content": "Tell me about your experience", "model": "groq"}'
# First request will be slow (RAG initialization)
# Subsequent requests will be fast
```

## Render Deployment
After this fix:
1. Commit and push changes
2. Render will deploy automatically
3. App should start and bind to port within 1 minute
4. First API request will take 3-5 minutes (RAG initialization)
5. All subsequent requests will be fast

## Manual Data Ingestion
After successful deployment, run data ingestion:
```bash
# SSH into Render instance or run locally with production DATABASE_URL
python scripts/ingest_profile_data.py
```

This will populate the vector database with profile embeddings.

---
**Date**: May 7, 2026
**Status**: Ready for deployment
