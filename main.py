from fastapi import FastAPI
import uvicorn
from routes.sentiment_route import router as sentiment_router
from routes.tokenizer_route import router as tokenizer_router

app = FastAPI()

# Include routers for modular API structure
app.include_router(sentiment_router)
app.include_router(tokenizer_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
