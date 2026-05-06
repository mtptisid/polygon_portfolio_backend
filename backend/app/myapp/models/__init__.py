"""
Models package for HR Assistant and database models.
"""

# HR Assistant models
from .hr_models import (
    RecruiterInfo,
    HRSessionStart,
    HRChatRequest,
    HRChatResponse,
    ChatMessage,
    SessionAnalysis,
    SessionMetadata,
    RecruiterSession,
    EndSessionRequest,
    AdminLogin,
    AdminToken,
    SessionSummary,
    SessionFilters
)

# Database models (SQLAlchemy)
from .db_models import User, ChatConversation

__all__ = [
    # HR models
    "RecruiterInfo",
    "HRSessionStart",
    "HRChatRequest",
    "HRChatResponse",
    "ChatMessage",
    "SessionAnalysis",
    "SessionMetadata",
    "RecruiterSession",
    "EndSessionRequest",
    "AdminLogin",
    "AdminToken",
    "SessionSummary",
    "SessionFilters",
    # DB models
    "User",
    "ChatConversation"
]
