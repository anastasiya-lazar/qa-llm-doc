from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.core.api import router as qa_router

app = FastAPI(
    title="QA System API",
    description="API for processing documents and answering questions",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(qa_router, prefix="/api/v1", tags=["qa"])

@app.get("/")
async def root():
    return {"message": "Welcome to the QA System API"} 