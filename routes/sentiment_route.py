from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from models.sentiment_model import analyse_sentiment
from models.tokenizer_model import tokenize_text
import time

router = APIRouter()


class TextSection(BaseModel):
    id: str
    content: str


class TextInput(BaseModel):
    text: str


class SectionInput(BaseModel):
    sections: List[TextSection]


@router.post("/get-sentiment")
def tokenize_and_analyse(data: TextInput):
    """
    Receives raw text, tokenizes it into chunks with sliding windows,
    and returns sentiment analysis based on the average scores across chunks.
    """
    total_start_time = time.perf_counter()

    # Tokenize and chunk the text
    tokenized = tokenize_text(data.text)

    # Run sentiment analysis on the pre-chunked tokens
    sentiment = analyse_sentiment(
        tokenized["input_ids"], tokenized["attention_mask"]
    )

    total_elapsed = time.perf_counter() - total_start_time
    print(f"[Total Request] Time taken: {total_elapsed * 1000:.2f} ms")

    return sentiment


@router.post("/get-sentiment-sections")
def tokenize_and_analyse_json(request: dict):
    """
    Receives a JSON object containing a 'sections' array, analyses each section separately,
    and returns a list of sentiment results.
    """
    total_start_time = time.perf_counter()
    results = []

    # Extract the sections array from the request
    sections = request.get("sections", [])

    for section in sections:
        # Tokenize and analyze each section separately
        tokenized = tokenize_text(section["content"])
        sentiment = analyse_sentiment(
            tokenized["input_ids"], tokenized["attention_mask"]
        )

        # Append results in required format
        results.append({
            "id": section["id"],
            "label": sentiment.get("label", ""),
            "score": sentiment.get("score", 0.0),
        })

    total_elapsed = time.perf_counter() - total_start_time
    print(f"[Total Request] Time taken: {total_elapsed * 1000:.2f} ms")

    return results
