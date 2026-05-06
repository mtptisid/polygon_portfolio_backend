"""
Pydantic models for HR Assistant and Admin Panel.

This module defines all data models for:
- Recruiter information and sessions
- Chat messages and responses
- Session analysis
- Admin authentication
"""

from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import List, Optional, Literal
from datetime import datetime
from uuid import UUID


# ============================================================================
# HR Assistant Models
# ============================================================================

class RecruiterInfo(BaseModel):
    """Information about the recruiter starting a session."""
    
    name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Recruiter's full name"
    )
    email: EmailStr = Field(
        ...,
        description="Recruiter's email address"
    )
    company: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Company name"
    )
    role: Optional[str] = Field(
        None,
        max_length=100,
        description="Role being recruited for"
    )
    additional_notes: Optional[str] = Field(
        None,
        max_length=500,
        description="Additional notes or context"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "John Doe",
                "email": "john.doe@techcorp.com",
                "company": "Tech Corp",
                "role": "Senior AI Engineer",
                "additional_notes": "Looking for LangChain expertise"
            }
        }


class HRSessionStart(BaseModel):
    """Response when starting a new HR session."""
    
    session_id: str = Field(
        ...,
        description="Unique session identifier"
    )
    welcome_message: str = Field(
        ...,
        description="Welcome message from the AI assistant"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Session start timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "welcome_message": "Hello! I'm Siddharamayya's AI assistant...",
                "timestamp": "2026-05-06T10:25:00Z"
            }
        }


class HRChatRequest(BaseModel):
    """Request to send a message in an HR session."""
    
    session_id: str = Field(
        ...,
        description="Session identifier"
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Message content"
    )
    model: Literal["gemini", "groq"] = Field(
        default="gemini",
        description="LLM model to use"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "content": "What is his experience with LangChain?",
                "model": "gemini"
            }
        }


class HRChatResponse(BaseModel):
    """Response from the HR assistant."""
    
    message_id: str = Field(
        ...,
        description="Unique message identifier"
    )
    content: str = Field(
        ...,
        description="AI assistant response"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "message_id": "660e8400-e29b-41d4-a716-446655440001",
                "content": "Siddharamayya has extensive experience with LangChain...",
                "timestamp": "2026-05-06T10:30:05Z"
            }
        }


class ChatMessage(BaseModel):
    """A single message in the chat history."""
    
    role: Literal["user", "assistant"] = Field(
        ...,
        description="Message role"
    )
    content: str = Field(
        ...,
        description="Message content"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Message timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "Tell me about his LangChain experience",
                "timestamp": "2026-05-06T10:30:00Z"
            }
        }


class SessionAnalysis(BaseModel):
    """AI-generated analysis of a recruiter session."""
    
    conversation_summary: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="3-5 sentence summary of the conversation"
    )
    key_topics_discussed: List[str] = Field(
        ...,
        min_items=1,
        max_items=20,
        description="List of key topics discussed"
    )
    role_fit_analysis: Optional[str] = Field(
        None,
        max_length=1000,
        description="Analysis of candidate fit for the role (if role was discussed)"
    )
    interest_level: Literal["low", "medium", "high"] = Field(
        ...,
        description="Inferred interest level based on conversation depth"
    )
    recommended_next_steps: List[str] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="Recommended next steps for the recruiter"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "conversation_summary": "Recruiter from Tech Corp seeking Senior AI Engineer with LangChain expertise...",
                "key_topics_discussed": ["LangChain", "RAG", "MLOps", "Docker"],
                "role_fit_analysis": "Strong fit for Senior AI Engineer role...",
                "interest_level": "high",
                "recommended_next_steps": [
                    "Schedule technical interview",
                    "Review GitHub portfolio"
                ]
            }
        }


class SessionMetadata(BaseModel):
    """Metadata about a recruiter session."""
    
    start_time: datetime = Field(
        ...,
        description="Session start time"
    )
    end_time: datetime = Field(
        ...,
        description="Session end time"
    )
    session_duration_seconds: int = Field(
        ...,
        ge=0,
        description="Session duration in seconds"
    )
    total_messages: int = Field(
        ...,
        ge=0,
        description="Total number of messages exchanged"
    )
    model_used: str = Field(
        ...,
        description="Primary LLM model used"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "start_time": "2026-05-06T10:25:00Z",
                "end_time": "2026-05-06T10:45:00Z",
                "session_duration_seconds": 1200,
                "total_messages": 12,
                "model_used": "gemini"
            }
        }


class RecruiterSession(BaseModel):
    """Complete recruiter session with all data."""
    
    session_id: str = Field(
        ...,
        description="Unique session identifier"
    )
    recruiter_info: RecruiterInfo = Field(
        ...,
        description="Recruiter information"
    )
    chat_history: List[ChatMessage] = Field(
        default_factory=list,
        description="Complete chat history"
    )
    analysis: Optional[SessionAnalysis] = Field(
        None,
        description="Session analysis (generated at end)"
    )
    metadata: Optional[SessionMetadata] = Field(
        None,
        description="Session metadata (populated at end)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "recruiter_info": {
                    "name": "John Doe",
                    "email": "john@techcorp.com",
                    "company": "Tech Corp",
                    "role": "Senior AI Engineer"
                },
                "chat_history": [
                    {
                        "role": "user",
                        "content": "Tell me about his experience",
                        "timestamp": "2026-05-06T10:30:00Z"
                    }
                ],
                "analysis": {
                    "conversation_summary": "Recruiter interested in AI expertise...",
                    "key_topics_discussed": ["AI", "ML", "LangChain"],
                    "interest_level": "high",
                    "recommended_next_steps": ["Schedule interview"]
                },
                "metadata": {
                    "start_time": "2026-05-06T10:25:00Z",
                    "end_time": "2026-05-06T10:45:00Z",
                    "session_duration_seconds": 1200,
                    "total_messages": 12,
                    "model_used": "gemini"
                }
            }
        }


class EndSessionRequest(BaseModel):
    """Request to end an HR session."""
    
    session_id: str = Field(
        ...,
        description="Session identifier to end"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }


# ============================================================================
# Admin Panel Models
# ============================================================================

class AdminLogin(BaseModel):
    """Admin login request."""
    
    password: str = Field(
        ...,
        min_length=8,
        description="Admin password"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "password": "secure_admin_password"
            }
        }


class AdminToken(BaseModel):
    """Admin authentication token response."""
    
    access_token: str = Field(
        ...,
        description="JWT access token"
    )
    token_type: str = Field(
        default="bearer",
        description="Token type"
    )
    expires_at: datetime = Field(
        ...,
        description="Token expiration timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_at": "2026-05-07T10:25:00Z"
            }
        }


class SessionSummary(BaseModel):
    """Summary of a recruiter session for list view."""
    
    session_id: str = Field(
        ...,
        description="Session identifier"
    )
    recruiter_name: str = Field(
        ...,
        description="Recruiter name"
    )
    recruiter_email: str = Field(
        ...,
        description="Recruiter email"
    )
    company: str = Field(
        ...,
        description="Company name"
    )
    role: Optional[str] = Field(
        None,
        description="Role being recruited for"
    )
    start_time: datetime = Field(
        ...,
        description="Session start time"
    )
    session_duration_seconds: int = Field(
        ...,
        description="Session duration"
    )
    interest_level: str = Field(
        ...,
        description="Interest level (low/medium/high)"
    )
    total_messages: int = Field(
        ...,
        description="Total messages exchanged"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "recruiter_name": "John Doe",
                "recruiter_email": "john@techcorp.com",
                "company": "Tech Corp",
                "role": "Senior AI Engineer",
                "start_time": "2026-05-06T10:25:00Z",
                "session_duration_seconds": 1200,
                "interest_level": "high",
                "total_messages": 12
            }
        }


class SessionFilters(BaseModel):
    """Filters for session list queries."""
    
    company: Optional[str] = Field(
        None,
        description="Filter by company (partial match)"
    )
    role: Optional[str] = Field(
        None,
        description="Filter by role (partial match)"
    )
    start_date: Optional[datetime] = Field(
        None,
        description="Filter sessions after this date"
    )
    end_date: Optional[datetime] = Field(
        None,
        description="Filter sessions before this date"
    )
    interest_level: Optional[Literal["low", "medium", "high"]] = Field(
        None,
        description="Filter by interest level"
    )
    sort_by: Literal["date", "company", "interest_level", "duration"] = Field(
        default="date",
        description="Sort field"
    )
    sort_order: Literal["asc", "desc"] = Field(
        default="desc",
        description="Sort order"
    )
