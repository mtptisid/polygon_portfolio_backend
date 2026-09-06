from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Boolean, Text
from myapp.database import Base
from datetime import datetime
from sqlalchemy.orm import relationship

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String, unique=True)
    password = Column(String)
    conversations = relationship("ChatConversation", back_populates="user")


class ChatConversation(Base):
    __tablename__ = "chat_conversations"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    session_id = Column(Integer, nullable=False)
    title = Column(String)
    session_created_at = Column(DateTime, default=datetime.utcnow)
    content = Column(String)
    is_bot = Column(Boolean)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="conversations")


class RecruiterSessionDB(Base):
    """Persisted HR assistant recruiter session (replaces old local-disk JSON storage)."""
    __tablename__ = "recruiter_sessions"
    session_id = Column(String, primary_key=True)
    recruiter_name = Column(String, index=True)
    recruiter_email = Column(String, index=True)
    company = Column(String, index=True)
    role = Column(String, nullable=True)
    start_time = Column(DateTime, index=True, nullable=True)
    interest_level = Column(String, nullable=True, index=True)
    session_data = Column(Text, nullable=False)  # full RecruiterSession model, JSON-serialized
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)