from typing import List, Optional
from pydantic import BaseModel, EmailStr
from datetime import datetime

class User(BaseModel):
    name: str
    email: EmailStr
    password: str

class ShowUser(BaseModel):
    id: int
    name: str
    email: EmailStr
    class Config:
        from_attributes = True

# Update schema for user details
class UserUpdate(BaseModel):
    name: Optional[str] = None
    password: Optional[str] = None

class ShowallUser(BaseModel):
    id: int
    name: str
    email: EmailStr

    class Config:
        from_attributes = True

class Login(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None


class MessageResponse(BaseModel):
    content: str
    is_bot: bool
    session_id: int
    timestamp: datetime
    tool_used: Optional[str] = None

class ChatSessionResponse(BaseModel):
    chat_id: int
    title: str
    created_at: datetime
    messages: List[MessageResponse]

    class Config:
        from_attributes = True

class MessageCreate(BaseModel):
    content: str
    model: str
    session_id: Optional[int] = None
    tool: Optional[str] = None


class SessionMessage(BaseModel):
    content: str
    is_bot: bool
    timestamp: datetime
    tool_used: Optional[str] = None

class Session(BaseModel):
    session_id: int
    messages: List[SessionMessage]
    created_at: datetime

