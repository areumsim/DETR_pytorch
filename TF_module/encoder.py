import torch
import torch.nn as nn

from TF_module.attn import MHAttention
from einops import rearrange


class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        n_hidden = cfg["encoder_params"]["n_hidden"]
        d_model = cfg["d_model"]

        # self.maskedMultiHeadAttention = MHAttention(
        #     d_model=d_model, **cfg["encoder_params"]
        # )

        self.multiHeadAttention = MHAttention(d_model=d_model, **cfg["encoder_params"])
        self.normalization = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, n_hidden),
            nn.Dropout(cfg["drpoout_p"]),
            nn.Linear(n_hidden, d_model),
        )

        self.normalization2 = nn.LayerNorm(d_model)

    def forward(self, x, pos):
        x_pos = x + pos

        # x = x + self.multiHeadAttention(x, x, x)  # q, k, v
        x = x + self.multiHeadAttention(x_pos, x_pos, x)  # q, k, v
        # x = x + self.maskedMultiHeadAttention(x_pos, x_pos, x, mask)  # q, k, v

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

    def forward(self, x, pos):
        x = rearrange(x, "b c h w-> b (h w) c")
        pos = rearrange(pos, "b c h w-> b (h w) c")
        # mask = rearrange(mask, "b c h w-> b (h w) c")

        for encoder in self.encoder_list:
            x = encoder(x, pos)

        return x


if __name__ == "__main__":
    import yaml

    with open("C:/Users/sar10/code/config.yaml", "r") as stream:
        cfg = yaml.safe_load(stream)
    cfg = cfg["tf_model_params"]

    image_fetaure_pos = torch.randn((8, 256, 800))
    image_fetaure_pos = rearrange(image_fetaure_pos, "b c N -> b N c")

    multipleEncoder = MultipleEncoder(cfg)
    en = multipleEncoder(image_fetaure_pos)
    print(en)  # torch.Size([8, 800, 256])
