import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("KBLab/megatron-bert-large-swedish-cased-165k")
model = AutoModelForSequenceClassification.from_pretrained("KBLab/robust-swedish-sentiment-multiclass")

def analyse_text_basic(text: str):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    # Get model outputs
    outputs = model(**inputs)
    logits = outputs.logits
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=1)
    # Get highest score and predicted class index
    score, pred = torch.max(probs, dim=1)

    # Define label mapping (adjust if needed based on your model's configuration)
    labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    pred_index = int(pred.item())
    label = labels[pred_index] if pred_index < len(labels) else str(pred_index)

    return score.item(), label


