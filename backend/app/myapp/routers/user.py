from fastapi import APIRouter
from .. import database, schemas, models,  oauth2
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, status
from ..repository import user
from typing import List

router = APIRouter(
    prefix="/api/user",
    tags=['Users']
)

get_db = database.get_db

@router.get('/', response_model=schemas.ShowUser)
async def get_user(
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(oauth2.get_current_user)
):
    return user.showid(current_user.id, db)

@router.post('/create', response_model=schemas.ShowUser)
def create_user(request: schemas.User, db: Session = Depends(get_db)):
    return user.create(request, db)

@router.get('/all', response_model=List[schemas.ShowallUser])
def get_user(db: Session = Depends(get_db)):
    return user.show_all_user(db)

@router.get('/{id}', response_model=schemas.ShowUser)
def get_user(id: int, db: Session = Depends(get_db)):
    return user.showid(id, db)

@router.get('/email/{email}', response_model=schemas.ShowUser)
def get_user(email: str, db: Session = Depends(get_db)):
    return user.showemail(email, db)

@router.put("/update/{id}", response_model=schemas.ShowUser)
def update_user(id: int, request: schemas.UserUpdate, db: Session = Depends(get_db)):
    return user.update(id, request, db)