import torch
from torch.utils.data import Dataset, DataLoader


class ReviewDataset(Dataset):
    def __init__(self, input_texts, target_texts, tokenizer, max_len=256, max_target_len=128):
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, index):
        input_text = self.input_texts[index]
        target_text = self.target_texts[index]

        input_encoding = self.tokenizer(
            input_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        target_encoding = self.tokenizer(
            target_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_target_len,
            return_tensors="pt"
        )

        labels = target_encoding.input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_encoding.input_ids.flatten(),
            'attention_mask': input_encoding.attention_mask.flatten(),
            'labels': labels.flatten()
        }


def create_data_loaders(train_inputs, train_targets, val_inputs, val_targets, tokenizer, batch_size=4):
    train_dataset = ReviewDataset(train_inputs, train_targets, tokenizer)
    val_dataset = ReviewDataset(val_inputs, val_targets, tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
