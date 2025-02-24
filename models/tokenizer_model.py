import time
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "KBLab/megatron-bert-large-swedish-cased-165k")

MAX_LENGTH = 512
OVERLAP = 256


def tokenize_text(text):
    start_time = time.perf_counter()

    encoded = tokenizer(text, truncation=False)
    input_ids = encoded.input_ids if isinstance(
        encoded.input_ids[0], list) else [encoded.input_ids]

    token_chunks = []
    attention_mask_chunks = []

    for start in range(0, len(input_ids[0]), MAX_LENGTH - OVERLAP):
        end = min(start + MAX_LENGTH, len(input_ids[0]))
        token_chunks.append(input_ids[0][start:end])
        attention_mask_chunks.append([1] * (end - start))
        if end == len(input_ids[0]):
            break

    elapsed = time.perf_counter() - start_time
    print(f"[Tokenization] Time taken: {elapsed:.4f} seconds")

    return {
        "input_ids": token_chunks,
        "attention_mask": attention_mask_chunks
    }
