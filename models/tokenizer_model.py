import time
from transformers import AutoTokenizer

MAX_LENGTH = 512
# Adjust this ratio to change the overlap (e.g. 0.3 for 30%, 0.5 for 50%)
SLIDING_OVERLAP_RATIO = 0.5
OVERLAP = int(MAX_LENGTH * SLIDING_OVERLAP_RATIO)

# Load tokenizer once
tokenizer = AutoTokenizer.from_pretrained(
    "KBLab/megatron-bert-large-swedish-cased-165k")


def tokenize_text(text):
    """
    Tokenizes text and splits it into chunks of MAX_LENGTH tokens,
    using a sliding window with an overlap of OVERLAP tokens.
    """
    start_time = time.perf_counter()
    encoded = tokenizer(text, truncation=False)

    # Ensure we have a flat list of token IDs
    input_ids = encoded.input_ids if isinstance(
        encoded.input_ids[0], list) else [encoded.input_ids]
    total_tokens = len(input_ids[0])

    token_chunks = []
    attention_mask_chunks = []

    current_pos = 0
    while current_pos < total_tokens:
        end = min(current_pos + MAX_LENGTH, total_tokens)
        token_chunks.append(input_ids[0][current_pos:end])
        # Create a simple attention mask (1 for each token in the chunk)
        attention_mask_chunks.append([1] * (end - current_pos))

        if end == total_tokens:
            break
        current_pos += (MAX_LENGTH - OVERLAP)

    elapsed = time.perf_counter() - start_time
    print(f"[Tokenization] Time taken: {elapsed * 1000:.2f} ms")
    print(f"[Tokenization] Total number of tokens: {total_tokens}")

    return {
        "input_ids": token_chunks,
        "attention_mask": attention_mask_chunks
    }
