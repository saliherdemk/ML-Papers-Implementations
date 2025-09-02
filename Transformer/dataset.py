import random

import torch
from torch.utils.data import Dataset


def generate_random_dates(n=10000):
    max_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    dataset = set()
    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    for _ in range(n):
        y = random.randint(1000, 2025)
        m = random.randint(1, 12)

        if (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0):
            max_days[1] = 29
        else:
            max_days[1] = 28

        d = random.randint(1, max_days[m - 1])
        dataset.add((f"{y:04d}-{m:02d}-{d:02d}", f"{months[m - 1]} {d}, {y}"))

    return list(dataset)


dataset = generate_random_dates()


class DateDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_encoded, y_encoded = self.tokenizer.encode(self.data[idx])

        x_tensor = torch.tensor(x_encoded, dtype=torch.long)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long)

        return x_tensor, y_tensor
