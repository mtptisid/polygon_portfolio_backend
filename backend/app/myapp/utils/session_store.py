"""
Session storage for HR Assistant.

Handles persistent storage of recruiter sessions as JSON files.
Uses async I/O for non-blocking file operations.
"""

import json
import os
import logging
from pathlib import Path
from typing import List, Optional
import aiofiles
from datetime import datetime

from myapp.models.hr_models import RecruiterSession

logger = logging.getLogger(__name__)


class SessionStore:
    """
    Manages persistent storage of recruiter sessions.
    
    Sessions are stored as individual JSON files in the configured directory.
    Each file is named {session_id}.json and contains the complete session data.
    """
    
    def __init__(self, storage_path: str = "backend/app/data/recruiter_sessions/"):
        """
        Initialize the session store.
        
        Args:
            storage_path: Directory path for storing session files
        """
        self.storage_path = Path(storage_path)
        self._ensure_directory_exists()
        logger.info(f"SessionStore initialized with path: {self.storage_path}")
    
    def _ensure_directory_exists(self):
        """Create the storage directory if it doesn't exist."""
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Storage directory ensured: {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to create storage directory: {e}")
            raise
    
    def _get_session_path(self, session_id: str) -> Path:
        """
        Get the file path for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Path object for the session file
        """
        return self.storage_path / f"{session_id}.json"
    
    async def save_session(self, session: RecruiterSession) -> bool:
        """
        Save a session to a JSON file.
        
        Args:
            session: RecruiterSession object to save
            
        Returns:
            True if successful, False otherwise
        """
        session_path = self._get_session_path(session.session_id)
        
        try:
            # Convert Pydantic model to dict with datetime serialization
            session_dict = session.model_dump(mode='json')
            
            # Write to file with pretty printing
            async with aiofiles.open(session_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(session_dict, indent=2, ensure_ascii=False))
            
            logger.info(f"Session saved successfully: {session.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")
            return False
    
    async def load_session(self, session_id: str) -> Optional[RecruiterSession]:
        """
        Load a session from a JSON file.
        
        Args:
            session_id: Session identifier
            
        Returns:
            RecruiterSession object if found, None otherwise
        """
        session_path = self._get_session_path(session_id)
        
        if not session_path.exists():
            logger.warning(f"Session file not found: {session_id}")
            return None
        
        try:
            async with aiofiles.open(session_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                session_dict = json.loads(content)
            
            # Convert dict to Pydantic model
            session = RecruiterSession(**session_dict)
            logger.debug(f"Session loaded successfully: {session_id}")
            return session
            
        except json.JSONDecodeError as e:
            logger.error(f"Corrupted JSON file for session {session_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    async def list_sessions(self) -> List[RecruiterSession]:
        """
        List all sessions in the storage directory.
        
        Returns:
            List of RecruiterSession objects
        """
        sessions = []
        
        try:
            # Get all JSON files in the directory
            json_files = list(self.storage_path.glob("*.json"))
            logger.debug(f"Found {len(json_files)} session files")
            
            # Load each session
            for json_file in json_files:
                session_id = json_file.stem  # Filename without extension
                session = await self.load_session(session_id)
                
                if session:
                    sessions.append(session)
                else:
                    logger.warning(f"Skipping corrupted session file: {json_file}")
            
            logger.info(f"Loaded {len(sessions)} sessions")
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session file.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        session_path = self._get_session_path(session_id)
        
        if not session_path.exists():
            logger.warning(f"Session file not found for deletion: {session_id}")
            return False
        
        try:
            session_path.unlink()
            logger.info(f"Session deleted successfully: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    async def session_exists(self, session_id: str) -> bool:
        """
        Check if a session file exists.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session exists, False otherwise
        """
        session_path = self._get_session_path(session_id)
        return session_path.exists()
    
    async def get_session_count(self) -> int:
        """
        Get the total number of sessions.
        
        Returns:
            Number of session files
        """
        try:
            json_files = list(self.storage_path.glob("*.json"))
            return len(json_files)
        except Exception as e:
            logger.error(f"Failed to count sessions: {e}")
            return 0
    
    async def get_sessions_by_company(self, company: str) -> List[RecruiterSession]:
        """
        Get all sessions from a specific company.
        
        Args:
            company: Company name (case-insensitive partial match)
            
        Returns:
            List of matching RecruiterSession objects
        """
        all_sessions = await self.list_sessions()
        company_lower = company.lower()
        
        matching_sessions = [
            session for session in all_sessions
            if company_lower in session.recruiter_info.company.lower()
        ]
        
        logger.debug(f"Found {len(matching_sessions)} sessions for company: {company}")
        return matching_sessions
    
    async def get_sessions_by_date_range(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[RecruiterSession]:
        """
        Get sessions within a date range.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            List of matching RecruiterSession objects
        """
        all_sessions = await self.list_sessions()
        
        filtered_sessions = []
        for session in all_sessions:
            if not session.metadata:
                continue
            
            session_date = session.metadata.start_time
            
            # Check start date
            if start_date and session_date < start_date:
                continue
            
            # Check end date
            if end_date and session_date > end_date:
                continue
            
            filtered_sessions.append(session)
        
        logger.debug(f"Found {len(filtered_sessions)} sessions in date range")
        return filtered_sessions


# Global session store instance
_session_store: Optional[SessionStore] = None


def get_session_store(storage_path: Optional[str] = None) -> SessionStore:
    """
    Get the global session store instance.
    
    Args:
        storage_path: Optional custom storage path
        
    Returns:
        SessionStore instance
    """
    global _session_store
    
    if _session_store is None:
        if storage_path:
            _session_store = SessionStore(storage_path)
        else:
            # Use default path from environment or fallback
            default_path = os.getenv(
                "SESSION_STORAGE_PATH",
                "backend/app/data/recruiter_sessions/"
            )
            _session_store = SessionStore(default_path)
    
    return _session_store
