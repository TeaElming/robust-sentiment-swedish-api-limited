# sentiment_router.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import time

# Import the single `analyse_sentiment` function:
from models.combined_model import analyse_sentiment

router = APIRouter()

class TextSection(BaseModel):
    id: str
    content: str

class TextInput(BaseModel):
    text: str

class SectionInput(BaseModel):
    sections: List[TextSection]

@router.post("/get-sentiment")
def get_sentiment(data: TextInput):
    """
    Analyzes sentiment of a single text.
    """
    start_time = time.perf_counter()
    sentiment_result = analyse_sentiment(data.text)
    total_elapsed = (time.perf_counter() - start_time) * 1000
    print(f"[Total Request] Time taken: {total_elapsed:.2f} ms")
    return sentiment_result

@router.post("/get-sentiment-sections")
def get_sentiment_sections(request: SectionInput):
    """
    Analyzes sentiment for multiple sections (each possibly short or long).
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
    print(f"[Total Request] Time taken: {total_elapsed:.2f} ms")
    return results
