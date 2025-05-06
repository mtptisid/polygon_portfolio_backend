from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from .. import schemas, database, models, oauth2
from datetime import datetime
from typing import Optional
from ..services.ai import ai_manager

router = APIRouter(
    prefix="/api/ai_chat",
    tags=["Chat"],
    dependencies=[Depends(oauth2.get_current_user)]
)

@router.get("/home")
async def get_chat_sessions(
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(oauth2.get_current_user)
):
    name = current_user.name
    return {"data": {"message": f"Welcome to your Custom AI Application PolyGenAI - {name}",
                     "submsg": "Select the AI model to get started"}}-m 

@router.get("/history", response_model=list[schemas.ChatSessionResponse])
async def get_chat_sessions(
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(oauth2.get_current_user)
):
    """Get user's chat session history, including all messages"""
    
    sessions = db.query(
        models.ChatConversation.session_id,
        func.min(models.ChatConversation.title).label('title'),
        func.min(models.ChatConversation.session_created_at).label('created_at')
    ).filter(
        models.ChatConversation.user_id == current_user.id
    ).group_by(
        models.ChatConversation.session_id
    ).order_by(
        func.min(models.ChatConversation.session_created_at).desc()
    ).all()

    session_responses = []
    for session in sessions:
        messages = db.query(models.ChatConversation).filter(
            models.ChatConversation.user_id == current_user.id,
            models.ChatConversation.session_id == session.session_id
        ).order_by(models.ChatConversation.timestamp.asc()).all()

        session_responses.append(schemas.ChatSessionResponse(
            chat_id=session.session_id,
            title=session.title,
            created_at=session.created_at,
            messages=[
                schemas.MessageResponse(
                    content=m.content,
                    is_bot=m.is_bot,
                    session_id=m.session_id,
                    timestamp=m.timestamp
                ) for m in messages
            ]
        ))

    return session_responses

@router.post("/request", response_model=schemas.MessageResponse)
async def send_message(
    message: schemas.MessageCreate,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(oauth2.get_current_user)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="User not authenticated")

    if not message.content.strip():
        raise HTTPException(status_code=400, detail="Message content cannot be empty")

    session_id = message.session_id
    if not session_id:
        max_session = db.query(func.max(models.ChatConversation.session_id)).scalar()
        session_id = (max_session or 0) + 1

    # Fetch memory for chat
    past_messages = db.query(models.ChatConversation).filter_by(
        user_id=current_user.id,
        session_id=session_id
    ).order_by(models.ChatConversation.timestamp.asc()).all()

    chat_history = [
        {"role": "assistant" if msg.is_bot else "user", "content": msg.content}
        for msg in past_messages
    ]
    chat_history.append({"role": "user", "content": message.content})

    # Save user message
    save_message_to_db(db, current_user.id, message.content, session_id, is_bot=False)

    # AI response
    ai_response = await ai_manager.get_response(message.model, chat_history)

    # Save bot response
    bot_message = save_message_to_db(db, current_user.id, ai_response, session_id, is_bot=True)

    return schemas.MessageResponse(
        content=bot_message.content,
        is_bot=True,
        session_id=bot_message.session_id,
        timestamp=bot_message.timestamp
    )

@router.delete("/{chat_id}")
async def delete_chat_session(
    chat_id: int,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(oauth2.get_current_user)
):
    """Delete a specific chat session and all its messages"""
    # Verify the chat session exists and belongs to the user
    session_exists = db.query(models.ChatConversation).filter(
        models.ChatConversation.user_id == current_user.id,
        models.ChatConversation.session_id == chat_id
    ).first()

    if not session_exists:
        raise HTTPException(status_code=404, detail="Chat session not found")

    # Delete all messages in the session
    db.query(models.ChatConversation).filter(
        models.ChatConversation.user_id == current_user.id,
        models.ChatConversation.session_id == chat_id
    ).delete()

    db.commit()
    return {"message": "Chat session deleted successfully"}

def save_message_to_db(db, user_id, content, session_id, is_bot):
    is_new_session = not db.query(models.ChatConversation).filter_by(
        user_id=user_id,
        session_id=session_id
    ).first()

    message = models.ChatConversation(
        user_id=user_id,
        session_id=session_id,
        content=content,
        is_bot=is_bot,
        timestamp=datetime.utcnow(),
        title=content[:25] if is_new_session else None,
        session_created_at=datetime.utcnow() if is_new_session else None
    )

    db.add(message)
    db.commit()
    db.refresh(message)
    return message