from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel
from typing import List, Optional, Union
from datetime import datetime
from uuid import uuid4
import asyncio
import logging
import re
import json
from fastapi.responses import JSONResponse
from langchain_community.tools import DuckDuckGoSearchRun
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from myapp.services.ai import ai_manager
from myapp.services.embedding import EmbeddingService
from myapp.services.vector_store import VectorStoreManager
from myapp.services.rag_retriever import RAGRetriever
from myapp.core.config import get_settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/ai_chat",
    tags=["Chat"]
)

# Define Pydantic models
class MessageCreate(BaseModel):
    content: str
    model: str
    session_id: Optional[Union[str, int]] = None  # Allow str or int
    tool: Optional[str] = None

class MessageResponse(BaseModel):
    content: str
    is_bot: bool
    session_id: str
    timestamp: datetime
    tool_used: Optional[str] = None
    retrieval_mode: Optional[str] = "rag"  # "rag" or "fallback"

class SessionMessage(BaseModel):
    content: str
    is_bot: bool
    timestamp: datetime
    tool_used: Optional[str] = None

class Session(BaseModel):
    session_id: str
    messages: List[SessionMessage]
    created_at: datetime

# Initialize DuckDuckGo Search Run tool
search = DuckDuckGoSearchRun()

# In-memory session store (replace with database for production)
SESSIONS = {}

# Initialize RAG components
RAG_ENABLED = False
rag_retriever = None
try:
    settings = get_settings()
    logger.info("Initializing RAG components...")
    embedding_service = EmbeddingService(model_name=settings.EMBEDDING_MODEL)
    vector_store = VectorStoreManager(settings.DATABASE_URL)
    rag_retriever = RAGRetriever(vector_store, embedding_service)
    RAG_ENABLED = True
    logger.info("RAG components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG: {e}")
    logger.warning("RAG disabled - using fallback mode")
    RAG_ENABLED = False

# Fallback profile (minimal profile data for when RAG is unavailable)
FALLBACK_PROFILE = {
    "name": "Siddharamayya Mathapati",
    "email": "msidrm455@gmail.com",
    "phone": "+91 97406 71620",
    "experience_years": "5+",
    "current_role": "Senior AI Engineer at Maveric Systems",
    "key_skills": [
        "Agentic AI (LangChain, LangGraph, AutoGen, CrewAI, MCP)",
        "AI/ML (LLMs, RAG, fine-tuning with QLoRA)",
        "LLM Observability (LangSmith, OpenTelemetry)",
        "DevOps (Docker, Kubernetes, Ansible)",
        "Web Development (FastAPI, React 19, TypeScript)",
        "Cloud (AWS Bedrock, GCP, Azure)",
        "Vector Databases (MongoDB Atlas, FAISS, ChromaDB)"
    ],
    "availability": "Immediately available for new opportunities as of May 2026",
    "contact": {
        "email": "msidrm455@gmail.com",
        "phone": "+91 97406 71620",
        "github": "https://github.com/mtptisid",
        "portfolio": "https://siddharamayya.in",
        "linkedin": "https://linkedin.com/in/siddharamayya-mathapati"
    }
}

async def search_web(query: str, model: str = "groq") -> str:
    """Perform a web search using DuckDuckGo via LangChain without site restrictions for broader results."""
    full_query = query
    logger.info(f"Performing search with query: {full_query}")
  
    try:
        search_results = await asyncio.to_thread(search.run, full_query)
        if not search_results or "No good" in search_results:
            return "Web Search Results: Limited or no specific results found. Rely on general knowledge for explanation."
        formatted_results = "Web Search Results:\n"
        result_lines = search_results.split("\n")
        for line in result_lines:
            if line.strip():
                formatted_results += f"{line.strip()}\n"
        formatted_results += "\nNote: Format these URLs in Markdown as `[name](link)` in the final response."
        logger.info(f"Formatted search results: {formatted_results}")
        return formatted_results
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        return f"Web Search Results: Search failed: {str(e)}. Proceed with general knowledge."

def clean_text(text) -> str:
    """Clean text by removing excessive newlines and unwanted characters."""
    # Handle list responses (e.g. gemini-2.5-flash thinking blocks)
    if isinstance(text, list):
        text = "".join(
            part if isinstance(part, str) else part.get("text", "") if isinstance(part, dict) else str(part)
            for part in text
        )
    if not text or not isinstance(text, str):
        return "No response generated. Please try rephrasing your query."
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[\r\t]', '', text)
    text = text.strip()
    return text

_ROLE_MAP = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": AIMessage,
}


async def get_gemini_response(messages: list) -> str:
    lc_messages = [
        _ROLE_MAP.get(m["role"], HumanMessage)(content=m["content"])
        for m in messages
    ]

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    llm_with_tools = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_api_key).bind_tools([search])
    response = await llm_with_tools.ainvoke(lc_messages)

    if response.tool_calls:
        for tool_call in response.tool_calls:
            query = tool_call["args"]["query"]
            result = await asyncio.to_thread(search.run, query)
            lc_messages.append(AIMessage(content="", tool_calls=[tool_call]))
            lc_messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
        response = await llm_with_tools.ainvoke(lc_messages)

    return response.content if isinstance(response.content, str) else "".join(
        part if isinstance(part, str) else part.get("text", "") for part in response.content
    )

@router.post("/request", response_model=MessageResponse)
async def send_message(request: Request, message: MessageCreate):
    """Send a message to the selected AI model, optionally using tools."""
    raw_body = await request.body()
    logger.info(f"Raw request payload: {raw_body.decode('utf-8')}")
  
    if not message.content.strip():
        raise HTTPException(status_code=400, detail="Message content cannot be empty")
    
    session_id = str(message.session_id or uuid4())  # Cast to str
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "session_id": session_id,
            "messages": [],
            "created_at": datetime.utcnow()
        }
    
    user_message = SessionMessage(
        content=message.content,
        is_bot=False,
        timestamp=datetime.utcnow(),
        tool_used=None
    )
    SESSIONS[session_id]["messages"].append(user_message)
    
    # Retrieve context using RAG or fallback
    retrieval_mode = "fallback"
    context = ""
    
    if RAG_ENABLED and rag_retriever:
        try:
            logger.info("Attempting RAG retrieval...")
            settings = get_settings()
            context = rag_retriever.retrieve(
                query=message.content,
                top_k=settings.RETRIEVAL_TOP_K,
                timeout=settings.RETRIEVAL_TIMEOUT_MS / 1000.0  # Convert ms to seconds
            )
            retrieval_mode = "rag"
            logger.info(f"RAG retrieval successful (mode: {retrieval_mode})")
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            logger.warning("Falling back to basic profile")
            context = json.dumps(FALLBACK_PROFILE, indent=2)
            retrieval_mode = "fallback"
    else:
        logger.info("RAG not enabled, using fallback profile")
        context = json.dumps(FALLBACK_PROFILE, indent=2)
        retrieval_mode = "fallback"
    
    # Construct system prompt with retrieved context
    base_instructions = """You are Siddharamayya Mathapati's AI assistant, an expert on his professional background, technical expertise, projects, and career achievements.

**Response Style & Formatting**:
- **Always provide DETAILED, COMPREHENSIVE responses** - Don't be concise unless explicitly asked
- Use **Markdown format** with proper structure (headers, lists, emphasis)
- **Bold** key terms: names, roles, technologies, companies, achievements
- Use `-` for bulleted lists with proper indentation and spacing
- Format URLs as **[name](link)** (e.g., **[GitHub](https://github.com/mtptisid)**)
- Include specific examples, metrics, and technical details when available
- Wrap code snippets in triple backticks with language identifier

**Response Guidelines**:
1. **Technical Questions**: Provide in-depth explanations with:
   - Specific technologies, tools, and frameworks used
   - Real project examples and implementations
   - Quantifiable achievements (percentages, time saved, improvements)
   - Technical architecture and design patterns
   - Certifications and validations

2. **Career Questions**: Include:
   - Detailed role descriptions and responsibilities
   - Key projects and their impact
   - Technologies and methodologies used
   - Team collaboration and leadership examples
   - Duration, location, and context

3. **Skills Questions**: Elaborate on:
   - Proficiency levels with context
   - Practical applications in real projects
   - Related technologies and tools
   - Learning journey and certifications
   - Best practices and expertise areas

4. **Project Questions**: Provide:
   - Comprehensive project descriptions
   - Technical stack and architecture
   - Problem solved and solution approach
   - Key features and innovations
   - Links to repositories or demos
   - Impact and outcomes

5. **Personal Questions** (marriage, workout, hobbies, age, etc.):
   - Respond with **playful sarcasm** and humor
   - Redirect to professional topics
   - Keep it light and entertaining

**Tone**: Professional and knowledgeable for technical topics, playful for personal questions.

**Profile Information**:
"""
    
    # Build chat history context
    chat_history_text = ""
    if session_id in SESSIONS and SESSIONS[session_id]["messages"]:
        chat_history_text = "\n**Chat History**:\n"
        for msg in SESSIONS[session_id]["messages"][-10:]:  # Last 10 messages
            role = "Bot" if msg.is_bot else "User"
            chat_history_text += f"**{role}**: {msg.content[:200]}...\n" if len(msg.content) > 200 else f"**{role}**: {msg.content}\n"
    
    # Construct full system prompt
    system_prompt = f"{base_instructions}\n{context}\n{chat_history_text}\n\nUse the above information to answer questions about Siddharamayya's profile."
    
    if retrieval_mode == "fallback":
        system_prompt += "\n\n(Note: Using fallback mode due to database unavailability)"
    
    chat_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message.content},
    ]
    
    response_content = ""
    tool_used = None
    
    if message.tool == "SearchWeb":
        tool_used = "SearchWeb"
        search_result = await search_web(message.content, message.model)
        augmented_prompt = (
            f"{system_prompt}\n"
            f"**Web Search Results**:\n{search_result}\n"
            "Provide a detailed response in **Markdown**, using **[name](link)** for all URLs in the search results or elsewhere."
        )
        chat_history = [
            {"role": "system", "content": augmented_prompt},
            {"role": "user", "content": message.content},
        ]
    
    # Try the requested model first, with automatic fallback to Groq
    selected_model = message.model
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to get response from {selected_model} (attempt {attempt + 1}/{max_retries})")
            
            if selected_model == "gemini":
                response_content = await get_gemini_response(chat_history)
            else:
                response_content = await ai_manager.get_response(selected_model, chat_history)
            
            logger.info(f"Successfully got response from {selected_model}")
            break
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"{selected_model} error (attempt {attempt + 1}/{max_retries}): {error_msg}")
            
            # Check if it's a rate limit or service unavailable error
            is_rate_limit = "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower()
            is_unavailable = "503" in error_msg or "UNAVAILABLE" in error_msg or "high demand" in error_msg.lower()
            
            if (is_rate_limit or is_unavailable) and selected_model != "groq":
                # Fallback to Groq
                logger.warning(f"{selected_model} is unavailable (rate limit or high demand), falling back to Groq")
                selected_model = "groq"
                
                # Update system prompt to mention the fallback
                if "system" in chat_history[0]["role"]:
                    chat_history[0]["content"] += f"\n\n(Note: Using Groq as fallback due to {message.model} unavailability)"
                
                # Try Groq immediately without waiting
                continue
            
            # If we've exhausted retries or it's already Groq, return error message
            if attempt == max_retries - 1:
                response_content = (
                    f"**Apologies**, I encountered an issue while generating a response.\n\n"
                    f"**Error**: {error_msg[:200]}\n\n"
                    f"Please try again in a moment, or try using a different model (Groq is usually more reliable)."
                )
                break
            
            # Wait before retry (exponential backoff)
            await asyncio.sleep(2 ** attempt)
    
    response_content = clean_text(response_content)
    
    # Add a subtle note if we fell back to a different model
    if selected_model != message.model and response_content and not response_content.startswith("**Apologies**"):
        logger.info(f"Response generated using {selected_model} (fallback from {message.model})")
    
    bot_message = SessionMessage(
        content=response_content,
        is_bot=True,
        timestamp=datetime.utcnow(),
        tool_used=tool_used
    )
    SESSIONS[session_id]["messages"].append(bot_message)
    
    logger.info(f"Generated response: {response_content[:200]}...")  # Log snippet to debug
    
    return MessageResponse(
        content=response_content,
        is_bot=True,
        session_id=session_id,
        timestamp=datetime.utcnow(),
        tool_used=tool_used,
        retrieval_mode=retrieval_mode
    )

@router.get("/history", response_model=List[Session])
async def get_session_history():
    """Retrieve all session histories."""
    return [Session(**session) for session in SESSIONS.values()]

@router.get("/")
async def homestart():
    return Response(status_code=204)
