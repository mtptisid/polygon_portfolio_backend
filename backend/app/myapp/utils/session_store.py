"""
Session storage for HR Assistant.

Persists recruiter sessions to the Postgres database (recruiter_sessions table)
instead of local disk JSON files. Cloud Run containers have ephemeral,
per-instance filesystems, so file-based storage was silently losing sessions
on every redeploy/restart/scale event and wasn't shared across instances.
"""

import json
import logging
from typing import List, Optional
from datetime import datetime

from myapp.database import SessionLocal
from myapp.models.db_models import RecruiterSessionDB
from myapp.models.hr_models import RecruiterSession

logger = logging.getLogger(__name__)


class SessionStore:
    """Manages persistent storage of recruiter sessions in the database."""

    def __init__(self, storage_path: Optional[str] = None):
        # storage_path kept only for backward-compatible signature; sessions now live in Postgres.
        if storage_path:
            logger.info(f"SessionStore is DB-backed; ignoring storage_path={storage_path}")
        logger.info("SessionStore initialized (Postgres-backed)")

    async def save_session(self, session: RecruiterSession) -> bool:
        """
        Save (insert or update) a session row.

        Args:
            session: RecruiterSession object to save

        Returns:
            True if successful, False otherwise
        """
        db = SessionLocal()
        try:
            session_dict = session.model_dump(mode="json")
            session_json = json.dumps(session_dict, ensure_ascii=False)

            record = db.query(RecruiterSessionDB).filter(
                RecruiterSessionDB.session_id == session.session_id
            ).first()

            if record is None:
                record = RecruiterSessionDB(session_id=session.session_id)
                db.add(record)

            record.recruiter_name = session.recruiter_info.name
            record.recruiter_email = session.recruiter_info.email
            record.company = session.recruiter_info.company
            record.role = session.recruiter_info.role
            record.start_time = session.metadata.start_time if session.metadata else None
            record.interest_level = session.analysis.interest_level if session.analysis else None
            record.session_data = session_json
            record.updated_at = datetime.utcnow()

            db.commit()
            logger.info(f"Session saved successfully: {session.session_id}")
            return True

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to save session {session.session_id}: {e}")
            return False
        finally:
            db.close()

    async def load_session(self, session_id: str) -> Optional[RecruiterSession]:
        """
        Load a session by id.

        Args:
            session_id: Session identifier

        Returns:
            RecruiterSession object if found, None otherwise
        """
        db = SessionLocal()
        try:
            record = db.query(RecruiterSessionDB).filter(
                RecruiterSessionDB.session_id == session_id
            ).first()

            if not record:
                logger.warning(f"Session not found: {session_id}")
                return None

            session_dict = json.loads(record.session_data)
            session = RecruiterSession(**session_dict)
            logger.debug(f"Session loaded successfully: {session_id}")
            return session

        except json.JSONDecodeError as e:
            logger.error(f"Corrupted session data for {session_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
        finally:
            db.close()

    async def list_sessions(self) -> List[RecruiterSession]:
        """
        List all sessions.

        Returns:
            List of RecruiterSession objects
        """
        db = SessionLocal()
        try:
            records = db.query(RecruiterSessionDB).order_by(
                RecruiterSessionDB.start_time.desc()
            ).all()
            logger.debug(f"Found {len(records)} session rows")

            sessions = []
            for record in records:
                try:
                    session_dict = json.loads(record.session_data)
                    sessions.append(RecruiterSession(**session_dict))
                except Exception as e:
                    logger.warning(f"Skipping corrupted session record {record.session_id}: {e}")

            logger.info(f"Loaded {len(sessions)} sessions")
            return sessions

        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []
        finally:
            db.close()

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session identifier

        Returns:
            True if successful, False otherwise
        """
        db = SessionLocal()
        try:
            record = db.query(RecruiterSessionDB).filter(
                RecruiterSessionDB.session_id == session_id
            ).first()

            if not record:
                logger.warning(f"Session not found for deletion: {session_id}")
                return False

            db.delete(record)
            db.commit()
            logger.info(f"Session deleted successfully: {session_id}")
            return True

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
        finally:
            db.close()

    async def session_exists(self, session_id: str) -> bool:
        """
        Check if a session exists.

        Args:
            session_id: Session identifier

        Returns:
            True if session exists, False otherwise
        """
        db = SessionLocal()
        try:
            return db.query(RecruiterSessionDB.session_id).filter(
                RecruiterSessionDB.session_id == session_id
            ).first() is not None
        finally:
            db.close()

    async def get_session_count(self) -> int:
        """
        Get the total number of sessions.

        Returns:
            Number of sessions
        """
        db = SessionLocal()
        try:
            return db.query(RecruiterSessionDB).count()
        except Exception as e:
            logger.error(f"Failed to count sessions: {e}")
            return 0
        finally:
            db.close()

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

            if start_date and session_date < start_date:
                continue
            if end_date and session_date > end_date:
                continue

            filtered_sessions.append(session)

        logger.debug(f"Found {len(filtered_sessions)} sessions in date range")
        return filtered_sessions


# Global session store instance
_session_store: Optional[SessionStore] = None


def get_session_store(storage_path: Optional[str] = None) -> SessionStore:
    """
    Get the global session store instance (Postgres-backed).

    Args:
        storage_path: Unused, kept for backward compatibility

    Returns:
        SessionStore instance
    """
    global _session_store

    if _session_store is None:
        _session_store = SessionStore(storage_path)

    return _session_store
