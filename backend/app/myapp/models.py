from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Boolean
from .database import Base
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