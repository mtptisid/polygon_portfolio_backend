import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  
from myapp.models import db_models
from myapp.database import engine
from myapp.routers import user, authentication, ai_chat, contact, sendmail, hr_assistant, admin
from myapp.startup import startup_tasks

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Portfolio Backend API", version="1.0.0")

# Register startup event handler
@app.on_event("startup")
async def on_startup():
    """Run startup tasks when the application starts."""
    try:
        logger.info("Starting application...")
        
        # Create database tables
        logger.info("Creating database tables...")
        db_models.Base.metadata.create_all(engine)
        logger.info("Database tables created successfully")
        
        # Run startup tasks (RAG ingestion, etc.)
        await startup_tasks()
        
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        # Don't raise - let the app start anyway
        logger.warning("Application will continue with limited functionality")

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
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],    # Allows all methods
    allow_headers=["*"],    # Allows all headers
)

# Health check endpoint (must be before startup)
@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "status": "ok",
        "message": "Portfolio Backend API",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Render"""
    return {"status": "healthy"}

# Include routers
#app.include_router(authentication.router)
app.include_router(sendmail.router)
app.include_router(ai_chat.router)
app.include_router(contact.router)
app.include_router(hr_assistant.router)
app.include_router(admin.router)
#app.include_router(user.router)
