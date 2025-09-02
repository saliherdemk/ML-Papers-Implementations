class Tokenizer:
    def __init__(self):
        nums = [str(i) for i in range(10)]
        uppers = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
        lowers = [chr(i) for i in range(ord("a"), ord("z") + 1)]
        self.input_max = 10 + 2
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
        x = ["<sos>"] + list(x) + ["<eos>"]
        y = ["<sos>"] + list(y) + ["<eos>"]
        while len(x) != self.input_max:
            x.append("<pad>")
        while len(y) != self.output_max:
            y.append("<pad>")
        res_x = [self.tokens_to_ids[i] for i in x]
        res_y = [self.tokens_to_ids[i] for i in y]

        return (res_x, res_y)

    def decode(self, ids):
        res = [self.ids_to_tokens[str(i)] for i in ids]
        return "".join(res)
