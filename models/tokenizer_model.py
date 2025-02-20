from transformers import AutoTokenizer

# Load tokenizer at startup
tokenizer = AutoTokenizer.from_pretrained(
    "KBLab/megatron-bert-large-swedish-cased-165k")

MAX_LENGTH = 512
SLIDING_WINDOW = 0.5 # 50% overlap
OVERLAP = int(MAX_LENGTH * SLIDING_WINDOW) # 256 tokens

def tokenize_text(text):
    """
    Tokenizes text and applies sliding window segmentation if needed.
    """

    encoded = tokenizer(text, truncation=False)  # Tokenize full text properly

    # Extract input_ids safely
    input_ids = encoded.input_ids if isinstance(
        encoded.input_ids, list) else [encoded.input_ids]

    # Handle empty input case
    if not input_ids or len(input_ids[0]) == 0:
        return {"input_ids": [], "attention_mask": []}

    # Split into sliding windows if needed
    token_chunks = []
    attention_mask_chunks = []

    for start in range(0, len(input_ids[0]), MAX_LENGTH - OVERLAP):
        end = min(start + MAX_LENGTH, len(input_ids[0]))
        token_chunks.append(input_ids[0][start:end])
        attention_mask_chunks.append(
            [1] * (end - start))  # Full attention mask

        if end == len(input_ids[0]):
            break  # Stop when we reach the end

    return {
        "input_ids": token_chunks,
        "attention_mask": attention_mask_chunks
    }
