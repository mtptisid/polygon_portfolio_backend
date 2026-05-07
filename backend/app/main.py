import os
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  

# Configure logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log startup
logger.info("=" * 60)
logger.info("Starting Portfolio Backend API")
logger.info("=" * 60)

# Import models and database with error handling
try:
    from myapp.models import db_models
    from myapp.database import engine
    logger.info("✓ Database models imported successfully")
except Exception as e:
    logger.error(f"✗ Failed to import database models: {e}")
    db_models = None
    engine = None

# Import routers with error handling
try:
    logger.info("Importing routers...")
    from myapp.routers import user, authentication, ai_chat, contact, sendmail, hr_assistant, admin
    logger.info("✓ Routers imported successfully")
except Exception as e:
    logger.error(f"✗ Failed to import routers: {e}", exc_info=True)
    raise

# Import background initialization (disabled to prevent OOM on Render free tier)
# RAG will initialize lazily on first request instead
initialize_rag_background = None
logger.info("✓ Background RAG initialization disabled (lazy loading on first request)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Runs before server starts accepting requests and after it shuts down.
    """
    # STARTUP: Run before server starts
    logger.info("=" * 60)
    logger.info("Lifespan startup beginning...")
    logger.info("=" * 60)
    
    # Create database tables (fast, synchronous)
    if db_models and engine:
        try:
            logger.info("Creating database tables...")
            db_models.Base.metadata.create_all(engine)
            logger.info("✓ Database tables created successfully")
        except Exception as e:
            logger.error(f"✗ Database table creation failed: {e}")
    
    logger.info("=" * 60)
    logger.info("✓ Lifespan startup complete - server will bind to port now")
    logger.info("=" * 60)
    
    yield  # Server binds to port and starts accepting requests HERE
    
    logger.info("=" * 60)
    logger.info("Server is now running and accepting requests")
    logger.info("=" * 60)
    
    # SHUTDOWN: Cleanup tasks
    logger.info("Shutting down...")


logger.info("Creating FastAPI app...")
app = FastAPI(
    title="Portfolio Backend API",
    version="1.0.0",
    lifespan=lifespan
)
logger.info("✓ FastAPI app created")

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
logger.info("Including routers in app...")
#app.include_router(authentication.router)
app.include_router(sendmail.router)
app.include_router(ai_chat.router)
app.include_router(contact.router)
app.include_router(hr_assistant.router)
app.include_router(admin.router)
#app.include_router(user.router)
logger.info("✓ All routers included")
logger.info("=" * 60)
logger.info("✓ Application initialization complete")
logger.info("=" * 60)
