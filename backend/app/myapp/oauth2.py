from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from . import schemas, database, models, token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

def get_current_user(data: str = Depends(oauth2_scheme), 
                     db: Session = Depends(database.get_db)
                     ):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Verify and decode token
        payload = token.verify_token(data, credentials_exception)
        
        email = payload.email
        
        # Fetch user from database
        user = db.query(models.User).filter(models.User.email == email).first()
        if not user:
            raise credentials_exception
            
        return user
    except JWTError:
        raise credentials_exception