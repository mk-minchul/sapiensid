import torch
import torch.nn as nn

def preproc_patch_emb(backbone, emb_dim, dynamic_patch_base_size):
    # make patch embed to be linear layer
    assert hasattr(backbone, 'patch_embed'), 'backbone does not have patch_embed'
    del backbone.patch_embed

    # in_feature = dynamic_patch_base_size ** 2 * 3
    # backbone.patch_embed = nn.Linear(in_features=in_feature, out_features=emb_dim, bias=True)
    # backbone.patch_embed.bias.data = torch.zeros_like(backbone.patch_embed.bias.data)
    return backbone

def preproc_cls_token(backbone, emb_dim):
    if not hasattr(backbone, 'cls_token'):
        backbone.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
    if hasattr(backbone, 'mask_token'):
        del backbone.mask_token
    return backbone

def preproc_pos_emb(backbone, pos_embed_type, dynamic_patch_base_size, input_size):
    assert hasattr(backbone, 'pos_embed'), 'backbone does not have pos_embed'
    if pos_embed_type == 'learned':
        # assume it is learned
        dim = backbone.pos_embed.shape[2]
        pos_embedding_2d = posemb_sincos_2d(
            h = input_size[0] // dynamic_patch_base_size,
            w = input_size[1] // dynamic_patch_base_size,
            dim = dim,
        ).unsqueeze(0)
        cls_token = torch.zeros_like(pos_embedding_2d[:, 0:1, :])
        pos_embedding_2d = torch.cat([cls_token, pos_embedding_2d], dim=1)
        backbone.pos_embed.data = pos_embedding_2d
        backbone.pos_embed.requires_grad = True
    elif pos_embed_type == 'sincos_2d':
        print('Making 2D Sine Cosine Pos Embedding')
        dim = backbone.pos_embed.shape[2]
        pos_embedding_2d = posemb_sincos_2d(
            h = input_size[0] // dynamic_patch_base_size,
            w = input_size[1] // dynamic_patch_base_size,
            dim = dim,
        ).unsqueeze(0)
        cls_token = torch.zeros_like(pos_embedding_2d[:, 0:1, :])
        pos_embedding_2d = torch.cat([cls_token, pos_embedding_2d], dim=1)
        backbone.pos_embed.data = pos_embedding_2d
        backbone.pos_embed.requires_grad = False
    else:
        raise ValueError(f'Unknown pos_embed_type: {pos_embed_type}')
    
    return backbone


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)
