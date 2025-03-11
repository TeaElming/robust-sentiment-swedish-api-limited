# sentiment_routes.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import time

# Import the functions from separate model files:
from models.combined_model import analyse_sentiment
from models.basic_sentiment_model import analyse_text_basic, analyse_text_ultra_basic

router = APIRouter()


class TextSection(BaseModel):
    id: str
    content: str


class TextInput(BaseModel):
    text: str


class SectionInput(BaseModel):
    sections: List[TextSection]


@router.post("/get-sentiment-basic")
def get_sentiment_basic(data: TextInput):
    start_time = time.perf_counter()
    score, label = analyse_text_basic(data.text)
    total_elapsed = (time.perf_counter() - start_time) * 1000
    print(
        f"[Total Request - get-sentiment-basic] Time taken: {total_elapsed:.2f} ms")
    return {"score": score, "label": label}


@router.post("/get-sentiment-ultra-basic")
def get_sentiment_ultra_basic(data: TextInput):
    start_time = time.perf_counter()
    score, label = analyse_text_ultra_basic(data.text)
    total_elapsed = (time.perf_counter() - start_time) * 1000
    print(
        f"[Total Request - get-sentiment-ultra-basic] Time taken: {total_elapsed:.2f} ms")
    return {"score": score, "label": label}


@router.post("/get-sentiment")
def get_sentiment(data: TextInput):
    """
    Analyzes sentiment of a single text.
    """
    start_time = time.perf_counter()
    sentiment_result = analyse_sentiment(data.text)
    total_elapsed = (time.perf_counter() - start_time) * 1000
    print(
        f"[Total Request - get-sentiment] Time taken: {total_elapsed:.2f} ms")
    return sentiment_result


@router.post("/get-sentiment-sections")
def get_sentiment_sections(request: SectionInput):
    """
    Analyzes sentiment for multiple sections.
    """
    start_time = time.perf_counter()
    results = []
    for section in request.sections:
        sentiment = analyse_sentiment(section.content)
        results.append({
            "id": section.id,
            "label": sentiment["label"],
            "score": sentiment["score"]
        })
    total_elapsed = (time.perf_counter() - start_time) * 1000
    print(
        f"[Total Request - get-sentiment-sections] Time taken: {total_elapsed:.2f} ms")
    return results
