from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

MAX_LENGTH = 512
OVERLAP = 256  # 50% of 512

# Load model and tokenizer at startup (avoid reloading in each request)
tokenizer = AutoTokenizer.from_pretrained(
    "KBLab/megatron-bert-large-swedish-cased-165k")
model = AutoModelForSequenceClassification.from_pretrained(
    "KBLab/robust-swedish-sentiment-multiclass")
classifier = pipeline("sentiment-analysis", model=model,
                      tokenizer=tokenizer, return_all_scores=True)


def analyse_sentiment(input_ids, attention_mask):
    """
    Processes tokenized input with sliding window (50% overlap if length > 512 tokens).
    Returns a single sentiment label with the highest average score.
    """

    def chunk_scores(chunk_ids, chunk_mask):
        inputs = {
            "input_ids": torch.tensor([chunk_ids], dtype=torch.long),
            "attention_mask": torch.tensor([chunk_mask], dtype=torch.long)
        }

        # Ensure the classifier returns a list of dictionaries
        output = classifier(inputs)
        if isinstance(output, list) and output and isinstance(output[0], list) and all(isinstance(i, dict) for i in output[0]):
            return output[0]  # List of dicts, one per sentiment label
        return []

    length = len(input_ids)
    start = 0
    all_scores = {}

    while start < length:
        end = min(start + MAX_LENGTH, length)
        chunk_ids = input_ids[start:end]
        chunk_mask = attention_mask[start:end]

        # Get sentiment distribution
        scores = chunk_scores(chunk_ids, chunk_mask)

        # Accumulate scores by label
        for s in scores:
            label = s.get("label", "UNKNOWN")  # Safe extraction
            score = float(s.get("score", 0.0))  # Ensure it's a float
            all_scores[label] = all_scores.get(label, 0.0) + score

        # Move forward with overlap
        if end == length:
            break
        start += (MAX_LENGTH - OVERLAP)

    # Avoid division by zero
    n_chunks = max(1, len(range(0, length, MAX_LENGTH - OVERLAP)))

    # Average scores by number of chunks
    for lbl in all_scores:
        all_scores[lbl] /= n_chunks

    # Select highest scoring label safely
    if all_scores:
        best_label = max(all_scores, key=lambda lbl: float(
            all_scores[lbl]))  # Ensure float values
        best_score = float(all_scores[best_label])  # Cast to float for safety
    else:
        best_label, best_score = "UNKNOWN", 0.0  # Handle empty input case

    return {"label": best_label, "score": best_score}
