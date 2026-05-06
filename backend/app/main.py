import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  
from myapp.models import db_models
from myapp.database import engine
from myapp.routers import user, authentication, ai_chat, contact, sendmail, hr_assistant, admin
from myapp.startup import startup_tasks

app = FastAPI()

# Register startup event handler
@app.on_event("startup")
async def on_startup():
    """Run startup tasks when the application starts."""
    await startup_tasks()

# Define the allowed origins explicitly
origins = [
      # For local development (uncommented and included)
    #"https://improved-spork-pjwxw7vj9545f6p4p-3000.app.github.dev/", 
    "https://siddharamayya.in", 
    "https://mtptisid.github.io",
    "https://portfolio.siddharamayya.in"
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
#db_models.Base.metadata.drop_all(bind=engine)
db_models.Base.metadata.create_all(engine)

# Include routers
#app.include_router(authentication.router)
app.include_router(sendmail.router)
app.include_router(ai_chat.router)
app.include_router(contact.router)
app.include_router(hr_assistant.router)
app.include_router(admin.router)
#app.include_router(user.router)
