from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
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
    total_start_time = time.perf_counter(
    )  # Adding this for evaluation prposes, can remove at a later stage
    # Step 1: Tokenization
    tokenized = tokenize_text(data.text)
    # Step 2: Sentiment Analysis
    sentiment = analyse_sentiment(
        tokenized["input_ids"], tokenized["attention_mask"])

    total_elapsed = time.perf_counter() - total_start_time
    print(f"[Total Request] Time taken: {total_elapsed * 1000:.2f} ms")

    return sentiment
