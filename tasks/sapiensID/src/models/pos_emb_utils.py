import math
import torch
from torch import nn
from torchvision import transforms

def is_vit_model(model):
    is_vit = 'vit' in model.config.yaml_path
    if is_vit:
        assert hasattr(model, 'net') and hasattr(model.net, 'pos_embed')
    return is_vit

def interpolate_pos_embed(model, input_size, model_cfg=None):
    if is_vit_model(model):
        print('Interpolating position encoding to match input size')
        # Interpolate position encoding to match the input size
        pos_embed = model.net.pos_embed.data
        h, w = input_size[1], input_size[2]
        patch_size = model.net.patch_embed.proj.kernel_size[0]
        if model_cfg.pos_embded_interpolate_method == 'resize':
            new_pos_embed = _interpolate_pos_embed(pos_embed, w, h, patch_size=patch_size)
        elif model_cfg.pos_embded_interpolate_method == 'pad':
            new_pos_embed = _pad_pos_embed(pos_embed, w, h, patch_size=patch_size)
        model.net.pos_embed.data = new_pos_embed
        model.net.patch_embed.img_size = (h, w)
        model.net.pos_embed.num_patches = (h // patch_size) * (w // patch_size)
        model.net.num_patches = model.net.pos_embed.num_patches

        # change transform
        def make_train_transform():
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((h, w), antialias=True),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            return transform
        def make_test_transform():
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((h, w), antialias=True),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            return transform
        return model, make_train_transform, make_test_transform



def _interpolate_pos_embed(pos_embed, w, h, patch_size):
    interpolate_antialias = False
    interpolate_offset = 0.1

    N = pos_embed.shape[1]
    desired_N  = (w // patch_size) * (h // patch_size)
    if desired_N == N and w == h:
        return pos_embed
    pos_embed = pos_embed.float()
    dim = pos_embed.shape[-1]
    w0 = w // patch_size
    h0 = h // patch_size
    M = int(math.sqrt(N))  # Recover the number of patches in each dimension
    assert N == M * M
    kwargs = {}
    if interpolate_offset:
        # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
        # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
        sx = float(w0 + interpolate_offset) / M
        sy = float(h0 + interpolate_offset) / M
        kwargs["scale_factor"] = (sx, sy)
    else:
        # Simply specify an output size instead of a scale factor
        kwargs["size"] = (w0, h0)
    reshaped_pos_embed = nn.functional.interpolate(
        pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
        mode="bicubic",
        antialias=interpolate_antialias,
        **kwargs,
    )
    assert (w0, h0) == reshaped_pos_embed.shape[-2:]
    reshaped_pos_embed = reshaped_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return reshaped_pos_embed.to(pos_embed.dtype)


def _pad_pos_embed(pos_embed, w, h, patch_size):
    N = pos_embed.shape[1]
    desired_N  = (w // patch_size) * (h // patch_size)
    if desired_N == N and w == h:
        return pos_embed
    pos_embed = pos_embed.float()
    dim = pos_embed.shape[-1]
    w0 = w // patch_size
    h0 = h // patch_size
    M = int(math.sqrt(N))  # Recover the number of patches in each dimension
    assert N == M * M
    assert w0 >= M
    assert h0 >= M
    pad = (w0 - M) // 2
    padded_pos_embed = nn.functional.pad(pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2), (pad, pad, pad, pad))
    assert (w0, h0) == padded_pos_embed.shape[-2:]
    padded_pos_embed = padded_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return padded_pos_embed.to(pos_embed.dtype)