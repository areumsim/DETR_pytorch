import torch
import torch.nn as nn

from input_embedding import Embedding_PosEncoding
from attn import MHAttention


class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        n_hidden = cfg["encoder_params"]["n_hidden"]
        d_model = cfg["d_model"]

        self.multiHeadAttention = MHAttention(d_model=d_model, **cfg["encoder_params"])
        self.normalization = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, n_hidden),
            nn.Dropout(cfg["drpoout_p"]),
            nn.Linear(n_hidden, d_model),
        )

        self.normalization2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.multiHeadAttention(x, x, x)
        x = self.normalization(x)

        x = x + self.ff(x)
        x = self.normalization2(x)
        return x


class MultipleEncoder(nn.Module):
    def __init__(self, cfg):
        super(MultipleEncoder, self).__init__()
        self.encoder_list = nn.ModuleList(
            [Encoder(cfg) for _ in range(cfg["encoder_params"]["n_iter"])]
        )

    def forward(self, x):
        for encoder in self.encoder_list:
            x = encoder(x)

        return x


if __name__ == "__main__":
    x = torch.tensor(
        [
            [1, 3, 4, 5, 6, 7, 34, 3, 34, 2, 123],
            [3, 34, 2, 123, 3, 34, 2, 123, 3, 34, 2],
        ]
    )

    embedding = Embedding_PosEncoding()
    e = embedding(x)
    print(e)

    # attention = Attention()
    # a = attention(e)
    # print(a)

    # multiHeadAttention = MHA()
    # mha = multiHeadAttention(e)
    # print(mha)

    # encoder = Encoder()
    # en1 = encoder(e)
    # print(en1)

    N = 3
    multipleEncoder = MultipleEncoder(N)
    en = multipleEncoder(e)
    print(en)
