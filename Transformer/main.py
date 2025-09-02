import torch
import torch.nn as nn
import torch.optim as optim
from dataset import DateDataset, generate_random_dates
from embedding_block import EmbeddingBlock
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from transformer_model import Encoder


def train(model, dataloader, vocab_size, num_epochs=10, lr=1e-4, device="cuda"):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            logits = model(src)

            loss = criterion(logits.view(-1, vocab_size), tgt.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")


def main():
    dataset = generate_random_dates()
    tokenizer = Tokenizer()

    date_dataset = DateDataset(dataset, tokenizer)
    dataloader = DataLoader(date_dataset, batch_size=5, shuffle=True)
    embed_dim = 16
    max_seq_len = 20

    model = Encoder(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
        pad_id=tokenizer.pad_token_id,
        num_heads=4,
        num_layers=2,
    )
    for src, tgt in dataloader:
        print(src.shape, model(src).shape)
        break


if __name__ == "__main__":
    main()
