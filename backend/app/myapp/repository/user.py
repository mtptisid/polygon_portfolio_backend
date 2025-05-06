from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from .. import models, schemas
from ..hashing import Hash
from pydantic import BaseModel, EmailStr
from typing import Optional


def user_exists(email: str, db: Session):
    return db.query(models.User).filter(models.User.email == email).first() is not None

def create(request: schemas.User, db: Session):
    if user_exists(request.email, db):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    new_user = models.User(
        name=request.name,
        email=request.email,
        password=Hash.bcrypt(request.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

def showid(id: int, db: Session):
    user = db.query(models.User).filter(models.User.id == id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with the id {id} is not available"
        )
    return user

def showemail(email: str, db: Session):
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with the email {email} is not available"
        )
    return user

def show_all_user(db: Session):
    users = db.query(models.User).all()
    if not users:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No users available"
        )
    return users

def authenticate_user(email: str, password: str, db: Session):
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user or not Hash.verify(user.password, password):
        return False
    return user

def update(id: int, request: schemas.UserUpdate, db: Session):
    user = db.query(models.User).filter(models.User.id == id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with the id {id} is not available"
        )
    
    # Update fields if provided
    if request.name is not None:
        user.name = request.name
    if request.password is not None:
        user.password = Hash.bcrypt(request.password)
    
    db.commit()
    db.refresh(user)
    return user