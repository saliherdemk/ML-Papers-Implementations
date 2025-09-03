import argparse

import torch
from tokenizer import Tokenizer
from transformer_model import Decoder, Encoder


def load_model(vocab_size, pad_id):
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
    encoder_model.load_state_dict(torch.load("./weights/encoder.pth"))
    decoder_model.load_state_dict(torch.load("./weights/decoder.pth"))

    device = "cpu"

    encoder_model = encoder_model.to(device)
    decoder_model = decoder_model.to(device)

    encoder_model.eval()
    decoder_model.eval()

    return encoder_model, decoder_model


def generate_date(dates):
    tokenizer = Tokenizer()
    encoder_model, decoder_model = load_model(
        tokenizer.vocab_size, tokenizer.pad_token_id
    )
    dates = dates.split(",")
    sample_num = len(dates)

    dates = [tokenizer.encode((date, []))[0] for date in dates]
    dates = torch.tensor(dates)
    max_seq_len = 20

    finished = torch.zeros(sample_num, dtype=torch.bool)

    with torch.no_grad():
        enc_out = encoder_model(dates)
        generated = torch.tensor([tokenizer.tokens_to_ids["<sos>"]])
        generated = generated.repeat(sample_num, 1)

        for _ in range(max_seq_len):
            dec_out = decoder_model(generated, enc_out)

            next_token_logits = dec_out[:, -1, :]

            next_token = torch.argmax(next_token_logits, dim=-1)
            next_token = torch.where(
                finished, torch.tensor(tokenizer.pad_token_id), next_token
            )

            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

            finished = finished | (next_token == tokenizer.tokens_to_ids["<eos>"])
            if finished.all():
                break
        print(tokenizer.decode(generated))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dates",
        required=True,
    )

    args = parser.parse_args()

    generate_date(args.dates)


if __name__ == "__main__":
    main()
