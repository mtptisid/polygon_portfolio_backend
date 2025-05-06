import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  
from myapp import models
from myapp.database import engine
from myapp.routers import user, authentication, ai_chat

app = FastAPI()

# Define the allowed origins explicitly
origins = [
      # For local development (uncommented and included)
    #"https://improved-spork-pjwxw7vj9545f6p4p-3000.app.github.dev/", 
    "http://localhost:5173", 
]

# Add CORS middleware with specific origins
app.add_middleware(
    CORSMiddleware,
    #allow_origins=["*"],  # Use the origins list instead of "*"
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],    # Allows all methods
    allow_headers=["*"],    # Allows all headers
)

# Database setup
#models.Base.metadata.drop_all(bind=engine)
models.Base.metadata.create_all(engine)

# Include routers
#app.include_router(authentication.router)
app.include_router(ai_chat.router)
#app.include_router(user.router)