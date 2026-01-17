from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.database import engine, Base
from app.routers import users, generation
from app.services.storage import storage

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Ensure local storage directories exist
    await storage.ensure_storage_exists()

    yield

    # Shutdown
    await engine.dispose()


app = FastAPI(
    title=settings.app_name,
    description="""
    ## Personalized Image Generation API

    This API allows you to:

    1. **Register users** by uploading 4-10 photos - the system extracts the best face
    2. **Generate personalized images** using just text prompts - the user's face is automatically included

    ### How it works:

    1. **Registration**: Upload multiple photos of a person. The system uses AI to detect faces,
       score them for quality, and extract the best one for future reference.

    2. **Generation**: Simply provide a text prompt like "enjoying at a beach" or "working out in gym".
       The system automatically retrieves the stored face and generates images of that specific person
       in the described scenario using Nano Banana AI.

    ### Example prompts:
    - "enjoying at a beach with sunset"
    - "working out in a modern gym"
    - "reading a book in a cozy cafe"
    - "hiking in beautiful mountains"
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving stored images
storage_path = Path(settings.storage_path)
storage_path.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=str(storage_path)), name="files")


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc) if settings.debug else "Internal server error",
            "type": type(exc).__name__,
        },
    )


# Include routers
app.include_router(users.router, prefix="/api/v1")
app.include_router(generation.router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": settings.app_name}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": settings.app_name,
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "register_user": "POST /api/v1/users/register",
            "get_user": "GET /api/v1/users/{user_id}",
            "generate_image": "POST /api/v1/generate",
            "generation_history": "GET /api/v1/generate/history/{user_id}",
        }
    }
