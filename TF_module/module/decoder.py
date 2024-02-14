import torch
import torch.nn as nn

from input_embedding import Embedding_PosEncoding
from attn import MHAttention

# input : b 11 64 (batch, len, d-model)


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()

        self.maskedMultiHeadAttention = MHAttention(
            d_model=cfg["d_model"], **cfg["decoder_params"]
        )
        self.normalization = nn.LayerNorm(cfg["d_model"])

        self.multiHeadAttention = MHAttention(
            d_model=cfg["d_model"], **cfg["decoder_params"]
        )
        self.normalization = nn.LayerNorm(cfg["d_model"])

        self.ff = nn.Sequential(
            nn.Linear(cfg["d_model"], cfg["decoder_params"]["n_hidden"]),
            self.get_activation(cfg["decoder_params"]["activation"]),
            nn.Dropout(0.2),
            nn.Linear(cfg["decoder_params"]["n_hidden"], cfg["d_model"]),
        )

        self.normalization2 = nn.LayerNorm(cfg["d_model"])

    def get_activation(self, activation):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        else:
            raise ValueError("Invalid activation function. Choose 'relu' or 'gelu'.")

    def forward(self, x, e):
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=-1) * -999
        x = x + self.maskedMultiHeadAttention(x, x, x, mask)
        x = self.normalization(x)

        x = x + self.multiHeadAttention(x, e, e)  # q, k, v
        x = self.normalization(x)

        x = x + self.ff(x)
        x = self.normalization2(x)
        return x


class MultipleDecoder(nn.Module):
    def __init__(self, cfg):
        super(MultipleDecoder, self).__init__()
        self.decoder_list = nn.ModuleList(
            [Decoder(cfg) for _ in range(cfg["decoder_params"]["n_iter"])]
        )

    def forward(self, x, e):
        for decoder in self.decoder_list:
            x = decoder(x, e)

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

    N = 3
    multipleDecoder = MultipleDecoder(N)
    en = multipleDecoder(e, e)
    print(en)
