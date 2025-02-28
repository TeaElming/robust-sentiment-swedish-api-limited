from fastapi import APIRouter
from pydantic import BaseModel
from models.sentiment_model import analyse_sentiment
from models.tokenizer_model import tokenize_text
import time

router = APIRouter()

class TextInput(BaseModel):
    text: str

@router.post("/get-sentiment")
def tokenize_and_analyse(data: TextInput):
    """
    Receives raw text, tokenizes it, and returns sentiment analysis in one request.
    """
    total_start_time = time.perf_counter()

    # Step 1: Tokenization (your custom logic from tokenizer_model)
    tokenized = tokenize_text(data.text)
    # Step 2: Sentiment Analysis
    sentiment = analyse_sentiment(
        tokenized["input_ids"], tokenized["attention_mask"]
    )

    total_elapsed = time.perf_counter() - total_start_time
    print(f"[Total Request] Time taken: {total_elapsed * 1000:.2f} ms")

    return sentiment
