from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import time

MAX_LENGTH = 512
OVERLAP = 256  # 50% of 512

# Load model and tokenizer at startup
tokenizer = AutoTokenizer.from_pretrained(
    "KBLab/megatron-bert-large-swedish-cased-165k")
model = AutoModelForSequenceClassification.from_pretrained(
    "KBLab/robust-swedish-sentiment-multiclass")
# Use top_k=None instead of return_all_scores=True
classifier = pipeline("sentiment-analysis", model=model,
                      tokenizer=tokenizer, top_k=None)


def analyse_sentiment(input_ids, attention_mask):
    """
    Processes tokenized input with sliding window (50% overlap if length > 512 tokens).
    Returns a single sentiment label with the highest average score.
    """
    start_time = time.perf_counter()

    def chunk_scores(chunk_ids, chunk_mask):
        # Ensure chunk_ids is a flat list before decoding
        if isinstance(chunk_ids[0], list):
            chunk_ids = [token for sublist in chunk_ids for token in sublist]

        # Convert tokenized input back into text
        text_input = tokenizer.decode(chunk_ids, skip_special_tokens=True)

        # Ensure classifier gets a proper string
        output = classifier(text_input)

        if isinstance(output, list) and len(output) > 0 and isinstance(output[0], list):
            return output[0]  # Extract scores
        return []

    length = len(input_ids)
    start = 0
    all_scores = {}

    num_chunks = 0  # Count processed chunks

    while start < length:
        end = min(start + MAX_LENGTH, length)
        chunk_ids = input_ids[start:end]
        chunk_mask = attention_mask[start:end]

        # Get sentiment distribution for the chunk
        scores = chunk_scores(chunk_ids, chunk_mask)

        # Accumulate scores by label
        for s in scores:
            label = s.get("label", "UNKNOWN")
            score = float(s.get("score", 0.0))
            all_scores[label] = all_scores.get(label, 0.0) + score

        num_chunks += 1

        # Move forward with overlap
        if end == length:
            break
        start += (MAX_LENGTH - OVERLAP)

    # Normalize scores across chunks
    if num_chunks > 0:
        for lbl in all_scores:
            all_scores[lbl] /= num_chunks  # Average across chunks

    # Select highest scoring label
    if all_scores:
        best_label = max(all_scores, key=lambda lbl: all_scores[lbl])
        best_score = all_scores[best_label]
    else:
        best_label, best_score = "UNKNOWN", 0.0  # Handle empty input case

    elapsed = time.perf_counter() - start_time
    print(f"[Sentiment Analysis] Time taken: {elapsed:.4f} seconds")

    return {"label": best_label, "score": best_score}
