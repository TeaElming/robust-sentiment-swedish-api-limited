from fastapi import APIRouter
from pydantic import BaseModel
from models.tokenizer_model import tokenize_text

router = APIRouter()

class TextInput(BaseModel):
    text: str

@router.post("/tokenize")
def tokenize(data: TextInput):
    """
    API route that receives raw text and returns tokenized input.
    """
    return tokenize_text(data.text)