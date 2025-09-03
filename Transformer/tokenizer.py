class Tokenizer:
    def __init__(self):
        nums = [str(i) for i in range(10)]
        uppers = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
        lowers = [chr(i) for i in range(ord("a"), ord("z") + 1)]
        self.input_max = 10
        self.output_max = 18 + 2
        self.vocab = nums + uppers + lowers + ["-", ",", " ", "<sos>", "<eos>", "<pad>"]
        self.vocab_size = len(self.vocab)
        self.tokens_to_ids = {str(self.vocab[i]): i for i in range(len(self.vocab))}
        self.ids_to_tokens = {
            str(i): str(self.vocab[i]) for i in range(len(self.vocab))
        }

        self.pad_token_id = self.tokens_to_ids["<pad>"]

    def encode(self, sample):
        x, y = sample
        x = list(x)
        y = list(y)

        x_seq = x + ["<pad>"] * (self.input_max - len(x))

        tgt_in = ["<sos>"] + y
        tgt_out = y + ["<eos>"]

        tgt_in += ["<pad>"] * (self.output_max - len(tgt_in))
        tgt_out += ["<pad>"] * (self.output_max - len(tgt_out))

        x_ids = [self.tokens_to_ids[i] for i in x_seq]
        tgt_in_ids = [self.tokens_to_ids[i] for i in tgt_in]
        tgt_out_ids = [self.tokens_to_ids[i] for i in tgt_out]

        return x_ids, tgt_in_ids, tgt_out_ids

    def decode(self, batch_ids):
        result = []
        for ids in batch_ids:
            res = [self.ids_to_tokens[str(i.item())] for i in ids]
            result.append("".join(res))
        return result
