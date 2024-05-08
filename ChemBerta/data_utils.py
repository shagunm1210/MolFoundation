import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, tokenizer, data, y_regression_values, max_input_length, max_target_length, device="cuda"):
        self.tokenizer = tokenizer
        self.data = data
        self.y_regression_values = y_regression_values
        self.device = device
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = str(self.data[idx])
        labels = str(self.data[idx])

        # tokenize data
        inputs = self.tokenizer(data, max_length=self.max_input_length, padding="max_length", truncation=True, return_tensors="pt")
        labels = self.tokenizer(labels, max_length=self.max_target_length, padding="max_length", truncation=True, return_tensors="pt")

        return {
            "input_ids": inputs["input_ids"].flatten().to(self.device),
            "attention_mask": inputs["attention_mask"].flatten().to(self.device),
            "labels": labels["input_ids"].flatten().to(self.device),
            "y_regression_values": torch.tensor(self.y_regression_values[idx]).to(self.device),
        }