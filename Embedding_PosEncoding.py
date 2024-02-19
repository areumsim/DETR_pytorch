import torch
import torch.nn as nn

import math
from einops import rearrange, repeat


class PositionalEncoding_TF(nn.Module):
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


class PositionalEncoding_learnable(nn.Module):
    def __init__(self, n_dim):
        super().__init__()
        self.embed_dim = int(n_dim / 2)
        self.row_embed = nn.Embedding(50, self.embed_dim)
        self.col_embed = nn.Embedding(50, self.embed_dim)

    def forward(self, x):
        h, w = x.size()[-2:]

        # self.col_embed = self.col_embed.to_device(x.device)
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_embed = self.col_embed(i)  # w, embed_dim
        y_embed = self.row_embed(j)  # h, embed_dim

        x_embed = repeat(x_embed, "w c -> c h w", h=h)
        y_embed = repeat(y_embed, "h c -> c h w", w=w)
        pos = torch.cat([x_embed, y_embed], dim=0)

        return pos


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p=0.2, max_len=50):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

        pos_encoding = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model)
        )

        # PE(pos, 2i) = sin(pos/10000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(position * division_term)
        # PE(pos, 2i + 1) = cos(pos/10000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(position * division_term)

        # Reshape pos_encoding for each position in the image
        row_embed = repeat(pos_encoding, "w c -> w h c", h=max_len)
        col_embed = repeat(pos_encoding, "h c -> w h c", w=max_len)

        self.pos_encoding = row_embed + col_embed

    def get_pos_encoding(self):
        return self.pos_encoding

    def forward(self, token_embedding: torch.tensor):
        self.pos_encoding = self.pos_encoding[
            : token_embedding.size(2), : token_embedding.size(3), :
        ]
        self.pos_encoding = repeat(
            self.pos_encoding, "h w c-> b c h w", b=token_embedding.size(0)
        ).to(token_embedding.device)

        return rearrange(
            self.dropout(token_embedding + self.pos_encoding), "b c h w-> b (h w) c"
        )


class PositionalEncoding_fix(nn.Module):
    def __init__(self, cfg, max_len=50):
        self.cfg = cfg
        # dim_model = cfg["n_dim"]
        dim_model = cfg["d_model"]

        super().__init__()
        pos_encoding = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model)
        )

        # PE(pos, 2i) = sin(pos/10000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(position * division_term)
        # PE(pos, 2i + 1) = cos(pos/10000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(position * division_term)

        # Reshape pos_encoding for each position in the image
        row_embed = repeat(pos_encoding, "w c -> w h c", h=max_len)
        col_embed = repeat(pos_encoding, "h c -> w h c", w=max_len)

        self.pos_encoding = row_embed + col_embed

    # def get_pos_encoding(self):
    #     return self.pos_encoding

    def forward(self, batch_size, img_w, img_h):
        pos_encoding = self.pos_encoding[:img_w, :img_h, :]
        pos_encoding = repeat(pos_encoding, "h w c-> b c h w", b=batch_size)
        # return rearrange(self.pos_encoding, "b c h w-> b (h w) c")
        return pos_encoding


if __name__ == "__main__":
    # ### PositionalEncoding_learnable
    n_dim = 256
    # positionalEncoding = PositionalEncoding_learnable(n_dim)
    positionalEncoding = PositionalEncoding(n_dim)

    # x = torch.randn((8, 256, 16, 16))
    # pos = positionalEncoding(x)
    # # image_fetaure_pos = image_fetaure +pos

    ### Fixed PositionalEncoding
    # n_dim = 256
    # positionalEncoding = PositionalEncoding(n_dim)

    image = torch.randn((8, 256, 20, 20))
    image_fetaure = torch.randn((8, 256, 400))  # torch.Size([8, 256, 400])

    image_fetaure_pos = positionalEncoding(image)
    image_fetaure_pos
    # pos = repeat(pos, "c h w -> b c h w", b=8)
    # pos = rearrange(pos, "b c h w-> b c (h w)")  # torch.Size([8, 256, 400])
    # pos = rearrange(pos, "c h w -> b c h w", b=8)

    # image_fetaure_pos = torch.concat(
    #     [image_fetaure, pos], dim=-1
    # )  # torch.Size([8, 256, 800])
