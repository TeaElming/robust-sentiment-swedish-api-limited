import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MAX_LENGTH = 512  # Maximum tokens per chunk

# Load tokenizer and model once (can be the same as in tokenization)
tokenizer = AutoTokenizer.from_pretrained(
    "KBLab/megatron-bert-large-swedish-cased-165k")
model = AutoModelForSequenceClassification.from_pretrained(
    "KBLab/robust-swedish-sentiment-multiclass")

# Apply dynamic quantization for faster CPU inference
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8)


def analyse_sentiment(token_chunks, mask_chunks):
    """
    Processes already-chunked input tokens in a single batched forward pass.
    Pads each chunk to MAX_LENGTH using vectorized operations,
    averages the probabilities over chunks,
    and returns the label with the highest average score.
    """
    start_time = time.perf_counter()

    # Ensure pad_token_id is an integer
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None or not isinstance(pad_token_id, int):
        pad_token_id = 0

    num_chunks = len(token_chunks)

    # Preallocate padded tensors using the integer pad_token_id
    input_ids_t = torch.full((num_chunks, MAX_LENGTH),
                             pad_token_id, dtype=torch.long)
    attention_mask_t = torch.zeros((num_chunks, MAX_LENGTH), dtype=torch.long)

    for i, (chunk, mask) in enumerate(zip(token_chunks, mask_chunks)):
        chunk_len = len(chunk)
        # Fill the corresponding slice with the actual tokens and masks
        input_ids_t[i, :chunk_len] = torch.tensor(chunk, dtype=torch.long)
        attention_mask_t[i, :chunk_len] = torch.tensor(mask, dtype=torch.long)

    with torch.no_grad():
        outputs = model(input_ids=input_ids_t, attention_mask=attention_mask_t)
        logits = outputs.logits      # shape: (num_chunks, num_labels)
        probs = torch.softmax(logits, dim=-1)

    # Average the probabilities across all chunks
    avg_probs = probs.mean(dim=0)
    best_idx = int(avg_probs.argmax().item())
    best_label = model.config.id2label[best_idx]
    best_score = float(avg_probs[best_idx].item())

    elapsed = time.perf_counter() - start_time
    print(f"[Sentiment Analysis] Time taken: {elapsed * 1000:.2f} ms")

    return {"label": best_label, "score": best_score}
