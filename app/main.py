from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.database import engine, Base
from app.routers import users, generation, video, tryon
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
    ## Personalized Image & Video Generation API

    This API allows you to:

    1. **Register users** by uploading 1-5 photos - stored as reference for all generation
    2. **Generate personalized images** using just text prompts - the user's face is automatically preserved
    3. **Generate personalized videos** from text prompts or existing images
    4. **Virtual Try-On** - upload any clothing photo and see how you'd look wearing it

    ### How it works:

    1. **Registration**: Upload photos of a person. These become the identity reference
       for all future generations.

    2. **Image Generation**: Provide a text prompt like "enjoying at a beach". The system
       sends ALL your reference photos to the AI for maximum face accuracy.

    3. **Video Generation**: Two modes:
       - **Text-to-Video**: Generate videos from text prompts with personalization
       - **Image-to-Video**: Animate an existing generated image into a video

    4. **Virtual Try-On**: Upload a clothing photo from any online store. The system
       generates a photorealistic image of YOU wearing that exact outfit — same face,
       same body, exact clothing design.

    ### Example prompts for images:
    - "enjoying at a beach with sunset"
    - "working out in a modern gym"
    - "reading a book in a cozy cafe"

    ### Example prompts for videos:
    - "walking gracefully on a beach at sunset"
    - "dancing in a modern studio"
    - "gentle smile and head turn" (for image-to-video)

    ### Virtual Try-On tips:
    - Use clear product photos from online stores (Shopify, Amazon, etc.)
    - Front-facing clothing shots work best
    - Add a description for better results: "navy blue formal suit with slim fit"
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
app.include_router(video.router, prefix="/api/v1")
app.include_router(tryon.router, prefix="/api/v1")


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
            "image_history": "GET /api/v1/generate/history/{user_id}",
            "generate_video_from_text": "POST /api/v1/video/from-text",
            "generate_video_from_image": "POST /api/v1/video/from-image",
            "video_history": "GET /api/v1/video/history/{user_id}",
            "virtual_tryon": "POST /api/v1/tryon",
            "tryon_history": "GET /api/v1/tryon/history/{user_id}",
        }
    }
