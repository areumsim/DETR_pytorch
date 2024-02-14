import torch
import torch.nn as nn

import math


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model
        )  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(
            token_embedding + self.pos_encoding[:, : token_embedding.size(1), :]
        )


class Embedding_PosEncoding(nn.Module):
    def __init__(self, cfg):
        super(Embedding_PosEncoding, self).__init__()
        self.embedding = nn.Embedding(cfg["n_words"], cfg["d_model"], max_norm=True)

        self.positionalEncoding = PositionalEncoding(
            cfg["d_model"], cfg["drpoout_p"], cfg["max_len"]
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.positionalEncoding(x)

        return x


if __name__ == "__main__":
    x = torch.tensor(
        [
            [1, 3, 4, 5, 6, 7, 34, 3, 34, 2, 123],
            [3, 34, 2, 123, 3, 34, 2, 123, 3, 34, 2],
        ]
    )

    d_model = 64
    n_words = 256
    drpoout_p = 0.2
    max_len = 100

    embedding = Embedding_PosEncoding(d_model, n_words, drpoout_p, max_len)
    e = embedding(x)
    print(e)
