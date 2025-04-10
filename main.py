from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from routes.sentiment_route import router as sentiment_router
from routes.tokenizer_route import router as tokenizer_router

app = FastAPI(root_path="/api/v2/sentimentis/") # TODO: Should be replaced with the actual root path when deploying, should have an ENV instead here

# Allow CORS (Cross-Origin Requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from all sources (use specific origins in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers for modular API structure
app.include_router(sentiment_router)
app.include_router(tokenizer_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
