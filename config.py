TOKENIZER_MODEL = "KBLab/megatron-bert-large-swedish-cased-165k"
SENTIMENT_MODEL = "KBLab/robust-swedish-sentiment-multiclass"

MAX_LENGTH = 512
SLIDING_WINDOW = 0.5 # 50% overlap
OVERLAP = int(MAX_LENGTH * SLIDING_WINDOW) # 256 tokens


API_PORT = 8000
TOKENIZER_PORT = 5001
