import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from typing import Union, Tuple, Dict

# Load the tokenizer and model (shared by all functions)
tokenizer = AutoTokenizer.from_pretrained(
    "KBLab/megatron-bert-large-swedish-cased-165k")
model = AutoModelForSequenceClassification.from_pretrained(
    "KBLab/robust-swedish-sentiment-multiclass")


def analyse_text_basic(text: str) -> tuple[float, str]:
    """
    Basic sentiment analysis:
    Tokenizes text, gets model logits, applies softmax, and returns the best score and label.
    """
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, padding=True)
    with torch.inference_mode():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=1)
    score, pred = torch.max(probs, dim=1)
    labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    pred_index = int(pred.item())
    label = labels[pred_index] if pred_index < len(labels) else str(pred_index)
    return score.item(), label


def analyse_text_ultra_basic(text: str) -> tuple[float, str]:
    """
    Ultra basic sentiment analysis using Hugging Face's pipeline.
    Returns a tuple (score, label).
    """
    classifier = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=512
    )
    result = classifier(text)
    if isinstance(result, list) and result:
        res = result[0]
        return res["score"], res["label"]
    return 0.0, "UNKNOWN"


def analyse_sentiment(text: str) -> dict:
    """
    Combined sentiment analysis function.
    (For now, this is a placeholder that simply calls analyse_text_basic.)
    """
    score, label = analyse_text_basic(text)
    return {"score": score, "label": label}


def analyse_multiple_ultra(texts: list[str]) -> list[dict]:
    """
    Processes multiple texts using the ultra basic method in parallel.
    Returns a list of dictionaries with keys "score" and "label".
    """
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(analyse_text_ultra_basic, texts))
    # Convert each result tuple to a dictionary
    return [{"score": score, "label": label} for score, label in results]


SentimentOutput = Union[Tuple[float, str], Dict[str, Union[float, str]]]

def analyse_long_ultra(text: str, window_size: int = 512, overlap: float = 0.5) -> SentimentOutput:
    # "window_size" is nominally 512, but we subtract a bit more
    # to avoid ever going above 512 once special tokens + re-tokenization are added.
    chunk_size = window_size - 4  # Subtract 4 instead of 2 to be safe

    tokens = tokenizer.encode(text, add_special_tokens=False)

    # If text is short enough, just do a single pass
    if len(tokens) <= chunk_size:
        return analyse_text_ultra_basic(text)

    stride = int(chunk_size * (1 - overlap))
    windows = []
    for i in range(0, len(tokens), stride):
        token_chunk = tokens[i:i + chunk_size]
        window_text = tokenizer.decode(
            token_chunk, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        windows.append(window_text)
        if i + chunk_size >= len(tokens):
            break

    results = analyse_multiple_ultra(windows)

    # Majority voting
    labels = [res["label"] for res in results]
    majority_label = Counter(labels).most_common(1)[0][0]
    majority_scores = [res["score"] for res in results if res["label"] == majority_label]
    overall_score = (sum(majority_scores) / len(majority_scores)
                     if majority_scores
                     else sum(res["score"] for res in results) / len(results))

    return {"score": overall_score, "label": majority_label}
