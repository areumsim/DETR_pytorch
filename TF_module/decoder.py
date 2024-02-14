import torch
import torch.nn as nn

# from input_embedding import Embedding_PosEncoding
from TF_module.attn import MHAttention
from einops import rearrange, repeat

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

    def forward(self, x, encoder_output, pos):
        # mask = torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=-1) * -999
        # x = x + self.maskedMultiHeadAttention(x, x, x, mask)
        x = x + self.multiHeadAttention(x, x, x)  # q, k, v
        x = self.normalization(x)

        # v : e_output / k : e_output + pos / q : x(origin)
        x = x + self.multiHeadAttention(encoder_output, x + pos, x)  # q, k, v
        x = self.normalization(x)

        x = x + self.ff(x)
        x = self.normalization2(x)
        return x

    # def forward(self, x):
    #     mask = torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=-1) * -999
    #     x = x + self.maskedMultiHeadAttention(x, x, x, mask)
    #     x = self.normalization(x)

    #     # v : e_output / k : e_output + posiotional / q : x(origin)
    #     x = x + self.multiHeadAttention(encoder_output, x, x)  # q, k, v
    #     x = self.normalization(x)

    #     x = x + self.ff(x)
    #     x = self.normalization2(x)
    #     return x


# class MultipleDecoder(nn.Module):
#     def __init__(self, cfg):
#         super(MultipleDecoder, self).__init__()
#         self.decoder_list = nn.ModuleList(
#             [Decoder(cfg) for _ in range(cfg["decoder_params"]["n_iter"])]
#         )

#     def forward(self, x, e):
#         for decoder in self.decoder_list:
#             x = decoder(x, e)

#         return x


class MultipleDecoder(nn.Module):
    def __init__(self, cfg):
        super(MultipleDecoder, self).__init__()
        self.x = nn.Parameter(torch.zeros(cfg["decoder_params"]["x_shape"]))

        self.decoder_list = nn.ModuleList(
            [Decoder(cfg) for _ in range(cfg["decoder_params"]["n_iter"])]
        )

        hidden_dim = cfg["d_model"]
        num_queries = cfg["decoder_params"]["n_queries"]
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

    def forward(self, en_output, pos):
        # x = rearrange(x, "b c h w-> b (h w) c")
        pos = rearrange(pos, "b c h w-> b (h w) c")

        # obj_query = torch.zeros(
        #     (labels.shape[0], labels.shape[1], 256), device=en_output.device
        # )

        for decoder in self.decoder_list:
            self.x = decoder(self.obj_query, en_output, pos)

        return self.x


if __name__ == "__main__":
    import yaml

    with open("C:/Users/sar10/code/config.yaml", "r") as stream:
        cfg = yaml.safe_load(stream)
    cfg = cfg["tf_model_params"]

    encoder_output = torch.randn((8, 800, 256))

    N = 3
    multipleDecoder = MultipleDecoder(cfg)
    en = multipleDecoder(encoder_output, encoder_output)
    print(en)
