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

    def forward(self, tgt, obj_query, encoder_output, encoder_pos):
        # self multi-head-att
        x = self.multiHeadAttention(obj_query+tgt, obj_query+tgt, tgt)  # q, k, v
        q = self.normalization(x + tgt)

        # multi-head-att
        x = self.multiHeadAttention(q+obj_query, encoder_output + encoder_pos, encoder_output)  # q, k, v
        x = self.normalization(x+q)

        x = self.normalization2(x + self.ff(x))
        return x

class MultipleDecoder(nn.Module):
    def __init__(self, cfg):
        super(MultipleDecoder, self).__init__()
        obj_query_shape = [1, cfg['n_obj'], cfg['d_model']] 
        self.obj_query = nn.Parameter(torch.randn(obj_query_shape))

        self.decoder_list = nn.ModuleList(
            [Decoder(cfg) for _ in range(cfg["decoder_params"]["n_iter"])]
        )

    def forward(self, tgt, en_output, en_pos):
        en_pos = rearrange(en_pos, "b c h w-> b (h w) c")

        for decoder in self.decoder_list:
            tgt = decoder(tgt, self.obj_query, en_output, en_pos)
        
        return tgt


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
