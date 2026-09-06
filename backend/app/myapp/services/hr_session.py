"""
HR Session Management with Conversation Memory.

Manages active recruiter sessions with conversation memory.
"""

import logging
from typing import Dict, Optional
from datetime import datetime
from uuid import uuid4

from myapp.models.hr_models import (
    RecruiterInfo,
    RecruiterSession,
    ChatMessage,
    SessionMetadata
)

logger = logging.getLogger(__name__)


class HRSessionManager:
    """
    Manages active HR assistant sessions with conversation memory.
    
    Each session maintains:
    - Recruiter information
    - Conversation history for context
    - Chat history
    - Session metadata
    """
    
    def __init__(self, max_sessions: int = 100):
        """
        Initialize the session manager.
        
        Args:
            max_sessions: Maximum number of concurrent sessions
        """
        self.max_sessions = max_sessions
        self.active_sessions: Dict[str, dict] = {}
        self.email_to_session: Dict[str, str] = {}  # Map email to active session_id
        logger.info(f"HRSessionManager initialized (max sessions: {max_sessions})")
    
    def get_or_create_session(self, recruiter_info: RecruiterInfo) -> tuple[str, bool]:
        """
        Get existing session for email or create new one.
        
        Args:
            recruiter_info: Information about the recruiter
            
        Returns:
            Tuple of (session_id, is_new_session)
        """
        email = recruiter_info.email.lower()
        
        # Check if active session exists for this email
        if email in self.email_to_session:
            existing_session_id = self.email_to_session[email]
            
            # Verify session still exists (might have been removed due to limit)
            if existing_session_id in self.active_sessions:
                logger.info(
                    f"Reusing existing session for {recruiter_info.name} ({email}): {existing_session_id}"
                )
                return existing_session_id, False
            else:
                # Session was removed, clean up mapping
                del self.email_to_session[email]
        
        # Create new session
        session_id = self.create_session(recruiter_info)
        
        # Map email to session
        self.email_to_session[email] = session_id
        
        return session_id, True
    
    def create_session(self, recruiter_info: RecruiterInfo) -> str:
        """
        Create a new HR session.
        
        Args:
            recruiter_info: Information about the recruiter
            
        Returns:
            Session ID (UUID string)
        """
        # Check session limit
        if len(self.active_sessions) >= self.max_sessions:
            self._remove_oldest_session()
        
        # Generate unique session ID
        session_id = str(uuid4())
        
        # Create session object
        session = {
            "session_id": session_id,
            "recruiter_info": recruiter_info,
            "chat_history": [],
            "start_time": datetime.utcnow(),
            "model_used": "gemini",  # Default, can be updated
            "message_count": 0
        }
        
        self.active_sessions[session_id] = session
        
        logger.info(
            f"Session created: {session_id} "
            f"(recruiter: {recruiter_info.name}, company: {recruiter_info.company})"
        )
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[dict]:
        """
        Get an active session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session dict if found, None otherwise
        """
        session = self.active_sessions.get(session_id)
        
        if not session:
            logger.warning(f"Session not found: {session_id}")
        
        return session
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        model: Optional[str] = None
    ) -> bool:
        """
        Add a message to the session.
        
        Args:
            session_id: Session identifier
            role: Message role ("user" or "assistant")
            content: Message content
            model: LLM model used (optional, for tracking)
            
        Returns:
            True if successful, False if session not found
        """
        session = self.get_session(session_id)
        
        if not session:
            return False
        
        # Create chat message
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.utcnow()
        )
        
        # Add to chat history
        session["chat_history"].append(message)
        session["message_count"] += 1
        
        # Update model if provided
        if model:
            session["model_used"] = model
        
        logger.debug(f"Message added to session {session_id}: {role}")
        return True
    
    def get_chat_history(self, session_id: str) -> list:
        """
        Get the chat history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of ChatMessage objects
        """
        session = self.get_session(session_id)
        
        if not session:
            return []
        
        return session["chat_history"]
    
    def get_memory_context(self, session_id: str) -> str:
        """
        Get the conversation context from chat history.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Formatted conversation history string
        """
        session = self.get_session(session_id)
        
        if not session:
            return ""
        
        try:
            chat_history = session["chat_history"]
            
            # Format the history
            formatted_history = []
            for msg in chat_history:
                role = "User" if msg.role == "user" else "Assistant"
                formatted_history.append(f"{role}: {msg.content}")
            
            return "\n".join(formatted_history)
            
        except Exception as e:
            logger.error(f"Failed to get memory context: {e}")
            return ""
    
    def end_session(self, session_id: str) -> Optional[RecruiterSession]:
        """
        End a session and prepare it for storage.
        
        Args:
            session_id: Session identifier
            
        Returns:
            RecruiterSession object ready for storage, or None if session not found
        """
        session = self.get_session(session_id)
        
        if not session:
            return None
        
        # Calculate session metadata
        end_time = datetime.utcnow()
        start_time = session["start_time"]
        duration_seconds = int((end_time - start_time).total_seconds())
        
        metadata = SessionMetadata(
            start_time=start_time,
            end_time=end_time,
            session_duration_seconds=duration_seconds,
            total_messages=session["message_count"],
            model_used=session["model_used"]
        )
        
        # Create RecruiterSession object
        recruiter_session = RecruiterSession(
            session_id=session_id,
            recruiter_info=session["recruiter_info"],
            chat_history=session["chat_history"],
            analysis=None,  # Will be added by analysis service
            metadata=metadata
        )
        
        # Remove email mapping
        email = session["recruiter_info"].email.lower()
        if email in self.email_to_session and self.email_to_session[email] == session_id:
            del self.email_to_session[email]
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        logger.info(
            f"Session ended: {session_id} "
            f"(duration: {duration_seconds}s, messages: {session['message_count']})"
        )
        
        return recruiter_session
    
    def restore_session(self, recruiter_session: "RecruiterSession") -> str:
        """
        Rehydrate a persisted session back into active memory.

        Used when a session_id sent by the frontend isn't in this instance's
        memory (e.g. Cloud Run restarted or routed the request to a different
        instance) but was previously autosaved to the database.

        Args:
            recruiter_session: RecruiterSession loaded from storage

        Returns:
            The session_id (for convenience/chaining)
        """
        if len(self.active_sessions) >= self.max_sessions:
            self._remove_oldest_session()
        
        session_id = recruiter_session.session_id
        session = {
            "session_id": session_id,
            "recruiter_info": recruiter_session.recruiter_info,
            "chat_history": list(recruiter_session.chat_history),
            "start_time": (
                recruiter_session.metadata.start_time if recruiter_session.metadata else datetime.utcnow()
            ),
            "model_used": recruiter_session.metadata.model_used if recruiter_session.metadata else "gemini",
            "message_count": len(recruiter_session.chat_history),
        }
        
        self.active_sessions[session_id] = session
        
        email = recruiter_session.recruiter_info.email.lower()
        self.email_to_session[email] = session_id
        
        logger.info(f"Session restored from storage: {session_id}")
        return session_id
    
    def _remove_oldest_session(self):
        """Remove the oldest session to make room for a new one."""
        if not self.active_sessions:
            return
        
        # Find oldest session by start_time
        oldest_id = min(
            self.active_sessions.keys(),
            key=lambda sid: self.active_sessions[sid]["start_time"]
        )
        
        oldest_session = self.active_sessions[oldest_id]
        
        # Remove email mapping
        email = oldest_session["recruiter_info"].email.lower()
        if email in self.email_to_session and self.email_to_session[email] == oldest_id:
            del self.email_to_session[email]
        
        logger.warning(
            f"Removing oldest session due to limit: {oldest_id} "
            f"(recruiter: {oldest_session['recruiter_info'].name})"
        )
        
        del self.active_sessions[oldest_id]
    
    def get_active_session_count(self) -> int:
        """
        Get the number of active sessions.
        
        Returns:
            Number of active sessions
        """
        return len(self.active_sessions)
    
    def clear_all_sessions(self):
        """Clear all active sessions (for testing/maintenance)."""
        count = len(self.active_sessions)
        self.active_sessions.clear()
        self.email_to_session.clear()
        logger.warning(f"Cleared all {count} active sessions")


# Global session manager instance
_session_manager: Optional[HRSessionManager] = None


def get_session_manager(max_sessions: int = 100) -> HRSessionManager:
    """
    Get the global session manager instance.
    
    Args:
        max_sessions: Maximum concurrent sessions
        
    Returns:
        HRSessionManager instance
    """
    global _session_manager
    
    if _session_manager is None:
        _session_manager = HRSessionManager(max_sessions=max_sessions)
    
    return _session_manager
