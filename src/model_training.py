import torch
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from src.config import LEARNING_RATE, EPOCHS, DEVICE, MODEL_SAVE_PATH


def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    losses = []

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return np.mean(losses)


def eval_model(model, data_loader, device):
    model.eval()
    losses = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            losses.append(loss.item())

    return np.mean(losses)


def train_model(model, train_loader, val_loader, tokenizer):
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=10,
        num_training_steps=total_steps
    )

    for epoch in range(EPOCHS):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            DEVICE,
            scheduler
        )

        val_loss = eval_model(
            model,
            val_loader,
            DEVICE
        )

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"Train loss: {train_loss}")
        print(f"Validation loss: {val_loss}")

        # Save the model at each epoch
        model_save_path = f"{MODEL_SAVE_PATH}t5_model_epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(), model_save_path)
