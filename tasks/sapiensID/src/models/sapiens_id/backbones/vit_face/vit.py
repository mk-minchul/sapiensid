import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from typing import Optional, Callable


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VITBatchNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm1d(num_features=num_features)

    def forward(self, x):
        return self.bn(x)


class Attention(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = False,
                 qk_scale: Optional[None] = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, offset_bias=None, effective_count=None):

        batch_size, num_token, embed_dim = x.shape
        #qkv is [3,batch_size,num_heads,num_token, embed_dim//num_heads]
        qkv = self.qkv(x).reshape(
            batch_size, num_token, 3, self.num_heads, embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q *= self.scale
        attn = (q @ k.transpose(-2, -1))

        # image relative position on keys
        if offset_bias is not None:
            attn += offset_bias

        if effective_count is not None:
            attn = attn + torch.log(effective_count)[:, None, None, :]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v

        x = out.transpose(1, 2).reshape(batch_size, num_token, embed_dim)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 num_patches: int,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: Optional[None] = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 act_layer: Callable = nn.ReLU6,
                 norm_layer: str = "ln",
                 patch_n: int = 144,
                 ):
        super().__init__()

        if norm_layer == "bn":
            self.norm1 = VITBatchNorm(num_features=num_patches)
            self.norm2 = VITBatchNorm(num_features=num_patches)
        elif norm_layer == "ln":
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.extra_gflops = (num_heads * patch_n * (dim//num_heads)*patch_n * 2) / (1000**3)

    def forward(self, x, offset_bias=None, effective_count=None):
        norm_x = self.norm1(x)
        attn_out = self.attn(norm_x, offset_bias=offset_bias, effective_count=effective_count)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=108, patch_size=9, in_channels=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        assert height == self.img_size[0] and width == self.img_size[1], \
            f"Input image size ({height}*{width}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer with auxiliary keypoint inputs for KP-RPE
    """

    def __init__(self,
                 img_size: int = 112,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: Optional[None] = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 num_patches: Optional[int] = None,
                 norm_layer: str = "ln",
                 mask_ratio = 0.1,
                 using_checkpoint = False,
                 ):
        super().__init__()
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim

        if num_patches is not None:
            self.patch_embed = nn.Identity()
        else:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
            num_patches = self.patch_embed.num_patches
        self.mask_ratio = mask_ratio
        self.using_checkpoint = using_checkpoint

        self.num_patches = num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        patch_n = (img_size//patch_size)**2
        self.blocks = nn.ModuleList(
            [
                Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                      num_patches=num_patches, patch_n=patch_n)
                for i in range(depth)]
        )
        self.extra_gflops = 0.0
        for _block in self.blocks:
            self.extra_gflops += _block.extra_gflops

        if norm_layer == "ln":
            self.norm = nn.LayerNorm(embed_dim)
        elif norm_layer == "bn":
            self.norm = VITBatchNorm(self.num_patches)

        # features head
        self.feature = None # do it in a different head

        if self.mask_ratio == 0:
            pass
        else:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            torch.nn.init.normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.num_heads = num_heads
        self.depth = depth



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def random_masking(self, x, mask_ratio=0.1):
        N, L, D = x.size()  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        index = ids_keep.unsqueeze(-1).repeat(1, 1, D)
        x_masked = torch.gather(x, dim=1, index=index)

        return x_masked, index, ids_restore

    def forward_blocks(self, x, offset_biases=None, effective_count=None):

        for block_idx, func in enumerate(self.blocks):
            if self.using_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(func, x, offset_biases[block_idx] if offset_biases is not None else None, effective_count=effective_count)
            else:
                x = func(x, offset_bias=offset_biases[block_idx] if offset_biases is not None else None, effective_count=effective_count)
        x = self.norm(x.float())
        return x

    def forward(self, x, keypoints=None):
        raise NotImplementedError('This model is only for forward blocks')
        return x


