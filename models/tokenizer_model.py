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
    print(f"Received text: {text}")  # Debugging

    encoded = tokenizer(text, truncation=False)
    print(f"Encoded output: {encoded}")  # Debugging

    # Extract input_ids safely
    input_ids = encoded.input_ids if isinstance(encoded.input_ids[0], list) else [encoded.input_ids]

    print(f"Processed input_ids: {input_ids}")  # Debugging

    # Handle empty input case
    if not input_ids or len(input_ids[0]) == 0:
        print("Tokenization failed: No input IDs found")  # Debugging
        return {"input_ids": [], "attention_mask": []}

    # Split into sliding windows if needed
    token_chunks = []
    attention_mask_chunks = []

    for start in range(0, len(input_ids[0]), MAX_LENGTH - OVERLAP):
        end = min(start + MAX_LENGTH, len(input_ids[0]))
        token_chunks.append(input_ids[0][start:end])
        attention_mask_chunks.append([1] * (end - start))

        if end == len(input_ids[0]):
            break

    print(f"Final token_chunks: {token_chunks}")  # Debugging
    print(f"Final attention_mask_chunks: {attention_mask_chunks}")  # Debugging

    return {
        "input_ids": token_chunks,
        "attention_mask": attention_mask_chunks
    }
