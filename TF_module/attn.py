import torch
import torch.nn as nn
from einops import rearrange


########### (v3) MHAttention : multihead + attention  ###########
# n_head 번 계산을 따로하지 않고,
# attention matrix를 n_head개 붙여서 한번에 계산하고 matrix를 n_head로 나눔
class MHAttention(nn.Module):
    def __init__(self, d_model, n_hidden, n_head, **kwargs):
        super(MHAttention, self).__init__()
        self.n_hidden = n_hidden

        # QW를 하려고 FC Layer , bias를 없애야함 wx+b 형태를 이용(b는 안씀)
        self.FC_Q = nn.Linear(d_model, n_hidden * n_head, bias=False)
        self.FC_K = nn.Linear(d_model, n_hidden * n_head, bias=False)
        self.FC_V = nn.Linear(d_model, n_hidden * n_head, bias=False)

        self.FC = nn.Linear((n_head * n_hidden), d_model, bias=False)

    # mat1 and mat2 shapes cannot be multiplied (7680x10 and 256x4096)
    def forward(self, q, k, v, mask=None):
        q = self.FC_Q(q)
        k = self.FC_K(k)
        v = self.FC_V(v)

        #   b  len n_hidden*n_head
        q = rearrange(q, "b l (c h) -> b l c h", c=self.n_hidden)
        k = rearrange(k, "b l (c h) -> b l c h", c=self.n_hidden)
        v = rearrange(v, "b l (c h) -> b l c h", c=self.n_hidden)

        x = torch.einsum("b l d h, b j d h -> b h l j", q, k) / (self.n_hidden) ** 0.5

        if mask is not None:
            mask = mask.to(x.device)
            x = x + mask

        x = torch.softmax(x, dim=-1)  # -1로 해야, 마지막 차원에서 합해서 1
        x = torch.einsum("b h l j, b j d h -> b l d h", x, v)
        x = rearrange(x, "b l d h -> b l (d h)")

        x = self.FC(x)

        return x


# #############################################################
# ########### (v2) mask 추가, encoder / decoder 공통 ###########
# class Attention(nn.Module):
#     def __init__(self, d_model, n_hidden):
#         super(Attention, self).__init__()
#         self.n_hidden = n_hidden

#         # QW를 하려고 FC Layer , bias를 없애야함 wx+b 형태를 이용(b는 안씀)
#         self.FC_Q = nn.Linear(d_model, n_hidden, bias=False)
#         self.FC_K = nn.Linear(d_model, n_hidden, bias=False)
#         self.FC_V = nn.Linear(d_model, n_hidden, bias=False)

#     def forward(self, q, k, v, mask=None):
#         q = self.FC_Q(q)
#         k = self.FC_K(k)
#         v = self.FC_V(v)

#         x = torch.einsum("b l d, b j d -> b l j", q, k) / (self.n_hidden) ** 0.5

#         if mask is not None:
#             x = x + mask

#         x = torch.softmax(x, dim=-1)  # -1로 해야, 마지막 차원에서 합해서 1
#         x = torch.einsum("b l j, b j d -> b l d", x, v)

#         return x


# class MHA(nn.Module):
#     def __init__(self, n_head, n_hidden, d_model, **kwargs):
#         super(MHA, self).__init__()
#         self.attention_list = nn.ModuleList(
#             [Attention(d_model, n_hidden) for i in range(n_head)]
#         )

#         # (self.n_head*n_hidden) : mha의 concat 길이/형태
#         # d_model : 원래 형태(model 차원으로) 출력
#         self.FC = nn.Linear((n_head * n_hidden), d_model, bias=False)

#     def forward(self, q, k, v, mask=None):
#         tmp = []
#         for attention in self.attention_list:
#             tmp.append(attention(q, k, v, mask))

#         x = torch.concat(tmp, -1)

#         x = self.FC(x)
#         return x
