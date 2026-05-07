"""
HR Assistant Router

Provides endpoints for recruiters to interact with the AI hiring assistant.
Includes session management, RAG-powered chat, and analysis generation.
"""

import logging
from fastapi import APIRouter, HTTPException
from uuid import uuid4
from datetime import datetime
import asyncio

from myapp.models.hr_models import (
    RecruiterInfo,
    HRSessionStart,
    HRChatRequest,
    HRChatResponse,
    EndSessionRequest,
    SessionAnalysis
)
from myapp.services.hr_session import get_session_manager
from myapp.services.hr_analysis import get_analysis_generator
from myapp.utils.session_store import get_session_store
from myapp.services.embedding import EmbeddingService
from myapp.services.vector_store import VectorStoreManager
from myapp.services.rag_retriever import RAGRetriever
from myapp.services.ai import ai_manager
from myapp.core.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/hr",
    tags=["HR Assistant"]
)

# Initialize components
session_manager = get_session_manager()
analysis_generator = get_analysis_generator()
session_store = get_session_store()

# Lazy RAG initialization (initialized on first use, not at import time)
_rag_retriever = None
_rag_initialized = False
_rag_init_lock = asyncio.Lock()
_rag_initializing = False

async def get_rag_retriever():
    """
    Lazy initialization of RAG components.
    Returns None if still initializing (background task running).
    """
    global _rag_retriever, _rag_initialized, _rag_initializing
    
    # If already initialized, return cached instance
    if _rag_initialized:
        return _rag_retriever
    
    # If background initialization is running, return None (will use fallback)
    if _rag_initializing:
        logger.info("RAG initialization in progress (background task)")
        return None
    
    # Use lock to prevent multiple simultaneous initializations
    async with _rag_init_lock:
        # Double-check after acquiring lock
        if _rag_initialized:
            return _rag_retriever
        
        if _rag_initializing:
            return None
        
        try:
            _rag_initializing = True
            settings = get_settings()
            logger.info("Initializing RAG components for HR Assistant on-demand...")
            
            # Initialize components
            embedding_service = EmbeddingService(model_name=settings.EMBEDDING_MODEL)
            vector_store = VectorStoreManager(settings.DATABASE_URL)
            _rag_retriever = RAGRetriever(vector_store, embedding_service)
            
            _rag_initialized = True
            _rag_initializing = False
            logger.info("✓ RAG components initialized successfully for HR Assistant")
            return _rag_retriever
            
        except Exception as e:
            logger.error(f"✗ Failed to initialize RAG for HR Assistant: {e}")
            _rag_initialized = True  # Mark as initialized to avoid retrying
            _rag_initializing = False
            _rag_retriever = None
            return None


@router.post("/start_session", response_model=HRSessionStart)
async def start_session(recruiter_info: RecruiterInfo):
    """
    Start a new HR assistant session.
    
    Creates a new session with the recruiter's information and initializes
    conversation memory.
    
    Args:
        recruiter_info: Recruiter information (name, email, company, role)
        
    Returns:
        Session ID and welcome message
    """
    try:
        # Create session
        session_id = session_manager.create_session(recruiter_info)
        
        # Generate welcome message
        welcome_message = f"""Hello! I'm Siddharamayya Mathapati's AI hiring assistant. I'm here to help you learn about his background, skills, and experience.

**About Me:**
I have comprehensive knowledge of Siddharamayya's:
- 5+ years of AI/ML and software engineering experience
- Expertise in LangChain, RAG, MLOps, and DevOps
- Projects, certifications, and technical skills
- Work history and achievements

**To Get Started:**
What role are you hiring for at {recruiter_info.company}? This will help me highlight the most relevant experience and skills for your needs.

Feel free to ask me anything about Siddharamayya's profile!"""
        
        logger.info(
            f"HR session started: {session_id} "
            f"(recruiter: {recruiter_info.name}, company: {recruiter_info.company})"
        )
        
        return HRSessionStart(
            session_id=session_id,
            welcome_message=welcome_message,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to start HR session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start session: {str(e)}"
        )


@router.post("/chat", response_model=HRChatResponse)
async def chat(request: HRChatRequest):
    """
    Send a message in an HR assistant session.
    
    Processes the recruiter's message using RAG retrieval and LLM,
    maintaining conversation context through LangChain memory.
    
    Args:
        request: Chat request with session_id, content, and model
        
    Returns:
        AI assistant response
    """
    try:
        # Get session
        session = session_manager.get_session(request.session_id)
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {request.session_id}"
            )
        
        # Add user message to session
        session_manager.add_message(
            request.session_id,
            "user",
            request.content,
            request.model
        )
        
        # Retrieve context using RAG
        context = ""
        rag_retriever = await get_rag_retriever()
        
        if rag_retriever:
            try:
                logger.info(f"Retrieving RAG context for: {request.content[:50]}...")
                settings = get_settings()
                context = rag_retriever.retrieve(
                    query=request.content,
                    top_k=settings.RETRIEVAL_TOP_K,
                    timeout=settings.RETRIEVAL_TIMEOUT_MS / 1000.0
                )
                logger.info("RAG context retrieved successfully")
            except Exception as e:
                logger.error(f"RAG retrieval failed: {e}")
                context = "Profile information temporarily unavailable."
        else:
            logger.warning("RAG not available, using limited context")
            context = "Profile information temporarily unavailable."
        
        # Get conversation history from memory
        memory_context = session_manager.get_memory_context(request.session_id)
        
        # Get recruiter info
        recruiter_info = session["recruiter_info"]
        
        # Build system prompt
        system_prompt = f"""You are a professional hiring assistant representing Siddharamayya Mathapati. You're helping {recruiter_info.name} from {recruiter_info.company} learn about the candidate.

**Your Role:**
- Be helpful, professional, and enthusiastic about the candidate
- Highlight relevant experience, skills, and achievements
- Provide specific examples with metrics and technologies
- Proactively ask about the role to provide tailored recommendations
- Suggest next steps (technical interview, portfolio review, etc.)

**Guidelines:**
- Use the profile information provided below
- Maintain conversation context
- When a role is mentioned, analyze fit and highlight relevant experience
- Be honest if information isn't available
- Format responses in Markdown with proper structure
- Use **bold** for emphasis on key terms

**Recruiter Context:**
- Company: {recruiter_info.company}
- Role: {recruiter_info.role or "Not yet specified"}
- Additional Notes: {recruiter_info.additional_notes or "None"}

**Profile Information:**
{context}

**Conversation History:**
{memory_context}

Respond to the recruiter's question professionally and helpfully."""
        
        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.content}
        ]
        
        # Call LLM with automatic fallback
        selected_model = request.model
        response_content = ""
        
        for attempt in range(2):
            try:
                logger.info(f"Calling {selected_model} for HR chat response")
                
                if selected_model == "gemini":
                    from myapp.routers.ai_chat import get_gemini_response
                    response_content = await get_gemini_response(messages)
                else:
                    response_content = await ai_manager.get_response(selected_model, messages)
                
                logger.info(f"Response generated successfully using {selected_model}")
                break
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"{selected_model} error: {error_msg}")
                
                # Check for rate limit or unavailability
                is_rate_limit = "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg
                is_unavailable = "503" in error_msg or "UNAVAILABLE" in error_msg
                
                if (is_rate_limit or is_unavailable) and selected_model != "groq":
                    logger.warning(f"Falling back to Groq due to {selected_model} unavailability")
                    selected_model = "groq"
                    continue
                
                # If all attempts failed
                if attempt == 1:
                    response_content = (
                        "I apologize, but I'm having trouble processing your request at the moment. "
                        "Please try again in a moment."
                    )
                    break
        
        # Add assistant response to session
        session_manager.add_message(
            request.session_id,
            "assistant",
            response_content,
            selected_model
        )
        
        # Generate message ID
        message_id = str(uuid4())
        
        logger.info(f"HR chat response sent (session: {request.session_id})")
        
        return HRChatResponse(
            message_id=message_id,
            content=response_content,
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process HR chat: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process message: {str(e)}"
        )


@router.post("/end_session", response_model=SessionAnalysis)
async def end_session(request: EndSessionRequest):
    """
    End an HR assistant session and generate analysis.
    
    Generates AI-powered analysis of the conversation including:
    - Conversation summary
    - Key topics discussed
    - Role fit analysis
    - Interest level
    - Recommended next steps
    
    Saves the complete session to storage.
    
    Args:
        request: End session request with session_id
        
    Returns:
        Session analysis
    """
    try:
        # Get session
        session = session_manager.get_session(request.session_id)
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {request.session_id}"
            )
        
        # End session and get RecruiterSession object
        recruiter_session = session_manager.end_session(request.session_id)
        
        if not recruiter_session:
            raise HTTPException(
                status_code=500,
                detail="Failed to end session"
            )
        
        # Generate analysis
        logger.info(f"Generating analysis for session: {request.session_id}")
        
        analysis = await analysis_generator.generate_analysis(
            chat_history=recruiter_session.chat_history,
            recruiter_info=recruiter_session.recruiter_info,
            model="groq"  # Use Groq for cost efficiency
        )
        
        # Add analysis to session
        recruiter_session.analysis = analysis
        
        # Save session to storage
        logger.info(f"Saving session to storage: {request.session_id}")
        save_success = await session_store.save_session(recruiter_session)
        
        if not save_success:
            logger.error(f"Failed to save session: {request.session_id}")
            # Don't fail the request, analysis is still returned
        
        logger.info(
            f"HR session ended: {request.session_id} "
            f"(interest: {analysis.interest_level}, "
            f"duration: {recruiter_session.metadata.session_duration_seconds}s)"
        )
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to end HR session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to end session: {str(e)}"
        )


@router.get("/")
async def hr_home():
    """Health check endpoint for HR Assistant."""
    # Check if RAG is initialized (without triggering initialization)
    rag_status = "initialized" if _rag_initialized and _rag_retriever else "not_initialized"
    
    return {
        "service": "HR Assistant",
        "status": "active",
        "rag_status": rag_status,
        "active_sessions": session_manager.get_active_session_count()
    }
