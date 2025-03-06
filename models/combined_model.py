import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MAX_LENGTH = 512
OVERLAP = 256

# If you have GPU, do:
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    "KBLab/megatron-bert-large-swedish-cased-165k")
model = AutoModelForSequenceClassification.from_pretrained(
    "KBLab/robust-swedish-sentiment-multiclass")

# Put model in eval mode and quantize
model.eval()
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8)

# If GPU is available, move to GPU
model.to(device)


def _analyse_sentiment_batched(token_chunks: list[list[int]], mask_chunks: list[list[int]]) -> dict:
    start_time = time.perf_counter()

    # Ensure pad_token_id is an integer
    pad_id = int(tokenizer.pad_token_id) if isinstance(
        tokenizer.pad_token_id, int) else 0

    num_chunks = len(token_chunks)

    # Prepare CPU Tensors first
    input_ids_t = torch.full((num_chunks, MAX_LENGTH),
                             pad_id, dtype=torch.long)
    attention_mask_t = torch.zeros((num_chunks, MAX_LENGTH), dtype=torch.long)

    for i, (chunk, mask) in enumerate(zip(token_chunks, mask_chunks)):
        chunk_len = len(chunk)
        input_ids_t[i, :chunk_len] = torch.tensor(chunk, dtype=torch.long)
        attention_mask_t[i, :chunk_len] = torch.tensor(mask, dtype=torch.long)

    # Move them to GPU if available
    input_ids_t = input_ids_t.to(device)
    attention_mask_t = attention_mask_t.to(device)

    # Use inference_mode for maximum speed
    with torch.inference_mode():
        outputs = model(input_ids=input_ids_t, attention_mask=attention_mask_t)
        probs = torch.softmax(outputs.logits, dim=-1)

    avg_probs = probs.mean(dim=0)
    best_idx = int(avg_probs.argmax().item())
    best_label = model.config.id2label[best_idx]
    best_score = float(avg_probs[best_idx].item())

    elapsed = (time.perf_counter() - start_time) * 1000
    print(f"[Sentiment Analysis] Time taken: {elapsed:.2f} ms")
    return {"label": best_label, "score": best_score}


def analyse_sentiment(text: str) -> dict:
    # Tokenize
    encoded = tokenizer(text, truncation=False, return_tensors="pt")
    input_ids_list = encoded.input_ids[0].tolist()

    if len(input_ids_list) <= MAX_LENGTH:
        return _analyse_sentiment_batched(
            token_chunks=[input_ids_list],
            mask_chunks=[[1]*len(input_ids_list)]
        )

    # If length > MAX_LENGTH, chunk
    token_chunks, mask_chunks = [], []
    start = 0
    while start < len(input_ids_list):
        end = min(start + MAX_LENGTH, len(input_ids_list))
        token_chunks.append(input_ids_list[start:end])
        mask_chunks.append([1]*(end - start))
        if end == len(input_ids_list):
            break
        start = end - OVERLAP

    return _analyse_sentiment_batched(token_chunks, mask_chunks)
