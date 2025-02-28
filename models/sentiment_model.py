from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import time

MAX_LENGTH = 512
OVERLAP = 256  # 50% of 512

# Load model and tokenizer at startup
tokenizer = AutoTokenizer.from_pretrained(
    "KBLab/megatron-bert-large-swedish-cased-165k"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "KBLab/robust-swedish-sentiment-multiclass"
)
# Use top_k=None to get a distribution across all labels
classifier = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    top_k=None
)


def analyse_sentiment(input_ids, attention_mask):
    """
    Processes tokenized input with a sliding window (50% overlap if >512 tokens).
    Returns a single sentiment label with the highest average score across chunks.
    """
    start_time = time.perf_counter()

    def chunk_scores(chunk_ids, chunk_mask):
        """
        Decodes the chunk of token IDs back to text, then calls the pipeline
        with `truncation=True, max_length=512` to avoid overflows.
        """
        # Flatten if it's nested
        if isinstance(chunk_ids[0], list):
            chunk_ids = [token for sublist in chunk_ids for token in sublist]

        # Convert tokenized IDs back into text
        text_input = tokenizer.decode(chunk_ids, skip_special_tokens=True)

        # Let the pipeline re-tokenize, but force truncation
        output = classifier(
            text_input,
            truncation=True,
            max_length=512,
        )
        # 'output' typically returns a list like:
        # [
        #   [{'label': 'POSITIVE', 'score': 0.98},
        #    {'label': 'NEGATIVE', 'score': 0.02}]
        # ]
        # or just [ { 'label': 'POSITIVE', 'score': 0.98 }, ... ]

        # If top_k=None, we usually see a nested list, so handle that:
        if isinstance(output, list) and len(output) > 0:
            # If it's a list-of-list of dicts, take the first inner list
            if isinstance(output[0], list):
                return output[0]  # e.g. [ {label,score}, {label,score}, ... ]
            else:
                # If pipeline returned a single list of dicts, just return it
                return output
        return []

    length = len(input_ids)
    start = 0
    all_scores = {}
    num_chunks = 0

    while start < length:
        end = min(start + MAX_LENGTH, length)
        chunk_ids = input_ids[start:end]
        chunk_mask = attention_mask[start:end]

        # Classify sentiment for this chunk
        scores = chunk_scores(chunk_ids, chunk_mask)

        # Accumulate by label
        for s in scores:
            label = s.get("label", "UNKNOWN")
            score = float(s.get("score", 0.0))
            all_scores[label] = all_scores.get(label, 0.0) + score

        num_chunks += 1

        if end == length:
            break
        start += (MAX_LENGTH - OVERLAP)  # Slide forward with overlap

    # Average scores across chunks
    if num_chunks > 0:
        for lbl in all_scores:
            all_scores[lbl] /= num_chunks

    # Pick highest average label
    if all_scores:
        best_label = max(all_scores, key=lambda lbl: all_scores[lbl])
        best_score = all_scores[best_label]
    else:
        best_label, best_score = "UNKNOWN", 0.0

    elapsed = time.perf_counter() - start_time
    print(f"[Sentiment Analysis] Time taken: {elapsed * 1000:.2f} ms")

    return {"label": best_label, "score": best_score}
