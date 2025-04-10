from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import time

# Import model functions from our separate models file.
from models.basic_sentiment_model import (
    analyse_text_ultra_basic,
    analyse_multiple_ultra,
    analyse_long_ultra
)

router = APIRouter()

class TextSection(BaseModel):
    id: str
    content: str

class TextInput(BaseModel):
    text: str

class SectionInput(BaseModel):
    sections: List[TextSection]


@router.post("/get-sentiment-ultra-basic")
def get_sentiment_ultra_basic(data: TextInput):
    score, label = analyse_text_ultra_basic(data.text)
    return {"score": score, "label": label}


@router.post("/get-sentiment-ultra-sections")
def get_sentiment_ultra_sections(request: SectionInput):
    """
    Analyzes sentiment for multiple sections using the ultra basic method in parallel.
    """
    texts = [section.content for section in request.sections]
    ultra_results = analyse_multiple_ultra(texts)
    combined_results = []
    for section, res in zip(request.sections, ultra_results):
        combined_results.append({
            "id": section.id,
            "score": res["score"],
            "label": res["label"]
        })
    return combined_results

@router.post("/get-sentiment-long-form")
def get_sentiment_long_form(data: TextInput):
    result = analyse_long_ultra(data.text)
    return result
