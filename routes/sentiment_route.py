from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from models.sentiment_model import analyse_sentiment

router = APIRouter()

class TokenizedInput(BaseModel):
    input_ids: List[int]          # Already tokenized input IDs
    attention_mask: List[int]     # Matching attention mask


@router.post("/sentiment")
def get_sentiment(token_data: TokenizedInput):
    """
    API route that receives tokenized text and returns sentiment analysis.
    """
    return analyse_sentiment(token_data.input_ids, token_data.attention_mask)
