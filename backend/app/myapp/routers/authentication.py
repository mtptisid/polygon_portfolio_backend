from fastapi import APIRouter, Depends, status, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from myapp import schemas, database, models, token
from myapp.schemas import Token
from myapp.hashing import Hash
from sqlalchemy.orm import Session
from ..repository import user

router = APIRouter(
    prefix="/api/auth",
    tags=['Authentication'])
get_db = database.get_db

@router.post("/register", response_model=Token)
async def register(user_data: schemas.User, db: Session = Depends(get_db)):
    # Check if user exists
    if user.user_exists(user_data.email, db):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Create user
    new_user = user.create(user_data, db)
    
    # Generate token using the actual new_user object
    access_token = token.create_access_token(data={"sub": new_user.email})
    return {"access_token": access_token, "token_type": "bearer"}


@router.post('/login')
def login(request: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    user = db.query(models.User).filter(
        models.User.email == request.username).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Invalid Credentials")
    if not Hash.verify(user.password, request.password):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Incorrect password")

    access_token = token.create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/verify-token/{token_id}")
async def verify_user_token(token_id: str):
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    token.verify_token(token_id, credentials_exception)
    return {"message": "Token is valid"}