import torch
import torch.nn as nn

import decoder
import encoder


class Transformer(nn.Module):
    def __init__(self, cfg):
        super(Transformer, self).__init__()
        self.input_embedding = encoder.Embedding_PosEncoding(cfg)
        self.multipleEncoder = encoder.MultipleEncoder(cfg)

        self.output_embedding = encoder.Embedding_PosEncoding(cfg)
        self.multipleDecoder = decoder.MultipleDecoder(cfg)

        self.linear = nn.Linear(cfg["d_model"], cfg["n_words"])

    def forward(self, x_in, x_out):
        encoder = self.input_embedding(x_in)
        encoder = self.multipleEncoder(encoder)

        x = self.output_embedding(x_out)
        x = self.multipleDecoder(x, encoder)

        x = self.linear(x)
        x = torch.softmax(x, dim=-1)
        return x


if __name__ == "__main__":
    import yaml

    x_in = torch.tensor(
        [
            [1, 3, 4, 5, 6, 7, 34, 3, 34, 2, 123],
            [3, 34, 2, 123, 3, 34, 2, 123, 3, 34, 2],
        ]
    )

    x_out = torch.tensor(
        [
            [1, 3, 4, 5, 6, 7, 34, 3, 34, 2, 123],
            [3, 34, 2, 123, 3, 34, 2, 123, 3, 34, 2],
        ]
    )

    with open("config.yaml", "r") as stream:
        cfg = yaml.safe_load(stream)

    transformer = Transformer(cfg["model_params"])
    tr = transformer(x_in, x_out)
    print(tr)

    ## 번역 문장 : predict sequece -> digit (original)
    print(torch.argmax(tr, dim=-1))


# N = 3
# tensor([[244,  97,  81,  70, 113,  72,  72,  59,  75,  75,  80],
#         [ 80, 149,  80, 111,  70,  72,  62,  72, 128, 224,  80]])

# N = 10
