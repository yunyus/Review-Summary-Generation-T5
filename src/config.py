import torch

# Configuration settings
MODEL_NAME = "Turkish-NLP/t5-efficient-base-turkish"
SPECIAL_TOKENS = {
    "additional_special_tokens": ["<REVIEW_SEP>", "<RATING_SEP>"]
}
DATA_FILE = "./data/summarized_reviews.csv"
MODEL_SAVE_PATH = "./checkpoints/"
MAX_LEN = 256
MAX_TARGET_LEN = 128
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 3e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
