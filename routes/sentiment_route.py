from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from models.sentiment_model import analyse_sentiment
from models.tokenizer_model import tokenize_text

router = APIRouter()


class TextInput(BaseModel):
    text: str


@router.post("/get-sentiment")
def tokenize_and_analyse(data: TextInput):
    """
    Receives raw text, tokenizes it, and returns sentiment analysis in one request.
    """
    print("PING PING PING")
    # Step 1: Tokenization
    tokenized = tokenize_text(data.text)
    # Step 2: Sentiment Analysis
    sentiment = analyse_sentiment(
        tokenized["input_ids"], tokenized["attention_mask"])
    return sentiment
