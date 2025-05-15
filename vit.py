import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class MultiHeadAttentionParallel(nn.Module):
    def __init__(self, embed_dim, num_heads, num_parallel):
        super(MultiHeadAttentionParallel, self).__init__()
        self.num_parallel = num_parallel
        self.attention_list = nn.ModuleList([nn.MultiheadAttention(embed_dim, num_heads) for _ in range(num_parallel)])

    def forward(self, x_parallel):
        return [self.attention_list[i](x, x, x)[0] for i, x in enumerate(x_parallel)]


class FeedForwardParallel(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_parallel):
        super(FeedForwardParallel, self).__init__()
        self.num_parallel = num_parallel
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, embed_dim)
            ) for _ in range(num_parallel)
        ])

    def forward(self, x_parallel):
        return [self.ff_layers[i](x) for i, x in enumerate(x_parallel)]


class TransformerEncoderLayerParallel(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, num_parallel):
        super(TransformerEncoderLayerParallel, self).__init__()
        self.self_attn = MultiHeadAttentionParallel(embed_dim, num_heads, num_parallel)
        self.feed_forward = FeedForwardParallel(embed_dim, hidden_dim, num_parallel)
        self.norm1 = ModuleParallel(nn.LayerNorm(embed_dim))
        self.norm2 = ModuleParallel(nn.LayerNorm(embed_dim))
        self.dropout = ModuleParallel(nn.Dropout(0.1))

    def forward(self, x_parallel):
        x_parallel2 = self.self_attn(x_parallel)
        x_parallel = [x + self.dropout(x2) for x, x2 in zip(x_parallel, x_parallel2)]
        x_parallel = self.norm1(x_parallel)

        x_parallel2 = self.feed_forward(x_parallel)
        x_parallel = [x + self.dropout(x2) for x, x2 in zip(x_parallel, x_parallel2)]
        x_parallel = self.norm2(x_parallel)

        return x_parallel


class TransformerEncoderParallel(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, num_layers, num_parallel):
        super(TransformerEncoderParallel, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayerParallel(embed_dim, num_heads, hidden_dim, num_parallel) for _ in range(num_layers)])

    def forward(self, x_parallel):
        for layer in self.layers:
            x_parallel = layer(x_parallel)
        return x_parallel


class VisionTransformerParallel(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, num_heads, hidden_dim, num_layers, num_classes, num_parallel):
        super(VisionTransformerParallel, self).__init__()
        self.num_parallel = num_parallel

        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = ModuleParallel(nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.encoder = TransformerEncoderParallel(embed_dim, num_heads, hidden_dim, num_layers, num_parallel)

        self.mlp_head = ModuleParallel(nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        ))

    def forward(self, x_parallel):
        x_parallel = [self.patch_embed(x).flatten(2).transpose(1, 2) for x in x_parallel]
        cls_tokens = [self.cls_token.expand(x.shape[0], -1, -1) for x in x_parallel]
        x_parallel = [torch.cat((cls_token, x), dim=1) for cls_token, x in zip(cls_tokens, x_parallel)]
        x_parallel = [x + self.pos_embed for x in x_parallel]

        x_parallel = self.encoder(x_parallel)
        x_parallel = [x[:, 0] for x in x_parallel]
        x_parallel = self.mlp_head(x_parallel)
        return x_parallel
