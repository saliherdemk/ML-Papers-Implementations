import torch
import torch.nn as nn
import torch.optim as optim
from dataset import DateDataset, generate_random_dates
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from transformer_model import Decoder, Encoder


def train(
    encoder_model, decoder_model, dataloader, vocab_size, num_epochs, lr, pad_id, device
):
    encoder_model = encoder_model.to(device)
    decoder_model = decoder_model.to(device)
    optimizer = torch.optim.Adam(
        list(encoder_model.parameters()) + list(decoder_model.parameters()), lr=lr
    )

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    for epoch in range(num_epochs):
        encoder_model.train()
        decoder_model.train()
        total_loss = 0

        for src, tgt_in, tgt_out in dataloader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

            enc_out = encoder_model(src)
            logits = decoder_model(tgt_in, enc_out)

            loss = criterion(logits.view(-1, vocab_size), tgt_out.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
        torch.save(encoder_model.state_dict(), "./weights/encoder.pth")
        torch.save(decoder_model.state_dict(), "./weights/decoder.pth")


def main():
    dataset = generate_random_dates()
    tokenizer = Tokenizer()

    date_dataset = DateDataset(dataset, tokenizer)
    dataloader = DataLoader(date_dataset, batch_size=5, shuffle=True)
    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim = 16
    max_seq_len = 20

    encoder_model = Encoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
        pad_id=pad_id,
        num_heads=4,
        num_layers=2,
    )

    decoder_model = Decoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
        pad_id=pad_id,
        num_heads=4,
        num_layers=2,
    )

    train(
        encoder_model,
        decoder_model,
        dataloader,
        vocab_size,
        num_epochs=50,
        lr=1e-4,
        pad_id=pad_id,
        device=device,
    )


if __name__ == "__main__":
    main()
