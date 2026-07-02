import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum, Tensor
from einops import rearrange, repeat
import math


class Rearrange(nn.Module):
    def __init__(self, pattern):
        super().__init__()
        self.pattern = pattern

    def forward(self, x):
        return rearrange(x, self.pattern)


class PatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels=2,
                 emb_size=40,
                 kernel_size=30,
                 stride=4,
                 pooling_size=8,
                 dropout_rate=0.3,
                 number_channel=22):
        super().__init__()
        self.cnn_module = nn.Sequential(
            nn.Conv1d(in_channels, emb_size, kernel_size, stride, padding=15),  # Temporal conv
            nn.BatchNorm1d(emb_size),
            nn.ELU(),
            nn.AvgPool1d(pooling_size),  # Pooling as slicing to obtain 'patch' along time dimension
            nn.Dropout(dropout_rate),
        )

        self.projection = nn.Sequential(
            Rearrange('b e (h) -> b h e'),  # [batch, embedding, time] -> [batch, patches, embedding]
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = x.squeeze(1)  # Remove spatial dimension
        x = self.cnn_module(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class ClassificationHead(nn.Sequential):
    def __init__(self, flatten_number, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flatten_number, n_classes)
        )

    def forward(self, x):
        out = self.fc(x)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x, **kwargs):
        x_input = x
        res = self.fn(x, **kwargs)
        out = self.layernorm(self.drop(res) + x_input)
        return out


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=4,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                MultiHeadAttention(emb_size, num_heads, drop_p),
            ), emb_size, drop_p),
            ResidualAdd(nn.Sequential(
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
            ), emb_size, drop_p)
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, heads, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size, heads) for _ in range(depth)])


class PositioinalEncoding(nn.Module):
    def __init__(self, embedding, length=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))

    def forward(self, x):  # x-> [batch, embedding, length]
        x = x + self.encoding[:, :x.shape[1], :].cuda()
        return self.dropout(x)


class PreBlock(nn.Module):
    def __init__(self, sampling_point):
        super().__init__()
        self.pool1 = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.AvgPool1d(kernel_size=13, stride=1, padding=6)
        self.pool3 = nn.AvgPool1d(kernel_size=7, stride=1, padding=3)
        self.ln_0 = nn.LayerNorm(sampling_point)
        self.ln_1 = nn.LayerNorm(sampling_point)

    def forward(self, x):
        x0 = x[:, 0, :, :].squeeze()
        x1 = x[:, 1, :, :].squeeze()

        x0 = self.pool1(x0)
        x0 = self.pool2(x0)
        x0 = self.pool3(x0)
        x0 = self.ln_0(x0)
        x0 = x0.unsqueeze(dim=1)

        x1 = self.pool1(x1)
        x1 = self.pool2(x1)
        x1 = self.pool3(x1)
        x1 = self.ln_1(x1)
        x1 = x1.unsqueeze(dim=1)

        x = torch.cat((x0, x1), 1)
        return x


class fNIRS_T(nn.Module):
    def __init__(self,
                 n_class: int,
                 sampling_point: int,
                 dim: int = 40,
                 depth: int = 6,
                 heads: int = 4,
                 mlp_dim: int = 256,
                 pool: str = 'cls',
                 dim_head: int = 64,
                 dropout: float = 0.,
                 emb_dropout: float = 0.):
        super().__init__()
        num_patches = 100

        self.preprocess = PreBlock(sampling_point)

        self.patch_embedding = PatchEmbedding(
            in_channels=2,
            emb_size=dim,
            kernel_size=30,
            stride=4,
            pooling_size=8,
            dropout_rate=0.3
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerEncoder(
            heads=heads,
            depth=depth,
            emb_size=dim
        )

        self.pool = pool
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_class)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.preprocess(x)
        b, c, h, w = x.shape

        x = x.reshape(b, c, -1)
        x = self.patch_embedding(x)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(x.shape[1])]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)