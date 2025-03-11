import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("KBLab/megatron-bert-large-swedish-cased-165k")
model = AutoModelForSequenceClassification.from_pretrained("KBLab/robust-swedish-sentiment-multiclass")

def analyse_text_basic(text: str) -> tuple[float, str]:
    # Tokenize the input text (with truncation and padding)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.inference_mode():
        outputs = model(**inputs)
    logits = outputs.logits
    # Convert logits to probabilities and pick the best label
    probs = F.softmax(logits, dim=1)
    score, pred = torch.max(probs, dim=1)
    labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    pred_index = int(pred.item())
    label = labels[pred_index] if pred_index < len(labels) else str(pred_index)
    return score.item(), label

def analyse_text_ultra_basic(text: str) -> tuple[float, str]:
    # Use the Hugging Face pipeline for sentiment-analysis.
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    result = classifier(text)
    # Ensure we return a tuple (score, label)
    if isinstance(result, list) and result:
        res = result[0]
        return res["score"], res["label"]
    return 0.0, "UNKNOWN"
