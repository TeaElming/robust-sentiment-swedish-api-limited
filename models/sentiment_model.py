import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MAX_LENGTH = 512  # Not used for additional chunking here

# Load tokenizer and model once (can be the same as in tokenization)
tokenizer = AutoTokenizer.from_pretrained(
    "KBLab/megatron-bert-large-swedish-cased-165k")
model = AutoModelForSequenceClassification.from_pretrained(
    "KBLab/robust-swedish-sentiment-multiclass")


def analyse_sentiment(token_chunks, mask_chunks):
    """
    Processes already-chunked input tokens in a single batched forward pass.
    Pads each chunk to MAX_LENGTH, averages the probabilities over chunks,
    and returns the label with the highest average score.
    """
    start_time = time.perf_counter()

    # Pad each chunk to MAX_LENGTH
    padded_token_chunks = []
    padded_mask_chunks = []
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    for chunk, mask in zip(token_chunks, mask_chunks):
        pad_len = MAX_LENGTH - len(chunk)
        # Pad token IDs with pad_token_id and mask with 0
        padded_token_chunks.append(chunk + [pad_token_id] * pad_len)
        padded_mask_chunks.append(mask + [0] * pad_len)

    # Convert lists of padded chunks to batched tensors
    input_ids_t = torch.tensor(padded_token_chunks, dtype=torch.long)
    attention_mask_t = torch.tensor(padded_mask_chunks, dtype=torch.long)

    with torch.no_grad():
        outputs = model(input_ids=input_ids_t, attention_mask=attention_mask_t)
        logits = outputs.logits      # shape: (num_chunks, num_labels)
        # shape: (num_chunks, num_labels)
        probs = torch.softmax(logits, dim=-1)

    # Average the probabilities across all chunks
    avg_probs = probs.mean(dim=0)  # shape: (num_labels,)
    best_idx = int(avg_probs.argmax().item())  # Cast explicitly to int
    best_label = model.config.id2label[best_idx]
    best_score = float(avg_probs[best_idx].item())

    elapsed = time.perf_counter() - start_time
    print(f"[Sentiment Analysis] Time taken: {elapsed * 1000:.2f} ms")

    return {"label": best_label, "score": best_score}
