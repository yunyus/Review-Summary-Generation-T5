import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from src.data_preparation import load_dataset, prepare_data, split_data
from src.dataset import create_data_loaders
from src.model_training import train_model
from src.config import MODEL_NAME, SPECIAL_TOKENS, BATCH_SIZE, DEVICE


def main():
    # Load and prepare the data
    df = load_dataset()
    input_texts, target_texts = prepare_data(df)
    train_inputs, val_inputs, train_targets, val_targets = split_data(
        input_texts, target_texts)

    # Initialize tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(DEVICE)

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_inputs, train_targets, val_inputs, val_targets, tokenizer, BATCH_SIZE)

    # Train the model
    train_model(model, train_loader, val_loader, tokenizer)


if __name__ == "__main__":
    main()
