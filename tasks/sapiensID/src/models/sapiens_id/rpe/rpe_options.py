import torch
import torch.nn as nn
from .RPE.KPRPE.kprpe_shared import get_rpe_config
from .RPE.KPRPE import relative_keypoints


def make_kprpe_shared(rpe_config, depth, num_heads):

    assert rpe_config.rpe_on == 'k'
    num_buckets = get_rpe_config(
        ratio=rpe_config.ratio,
        method=rpe_config.method,
        mode=rpe_config.mode,
        shared_head=rpe_config.shared_head,
        skip=1,
        rpe_on=rpe_config.rpe_on,
    )['rpe_k']['num_buckets']

    if rpe_config.ctx_type == 'rel_keypoint_splithead_unshared':
        keypoint_linear = nn.Linear(rpe_config.kp_dim * rpe_config.num_keypoints, 
                                    num_buckets * num_heads * depth)
        # init zero
        keypoint_linear.weight.data.zero_()
        keypoint_linear.bias.data.zero_()

    elif rpe_config.ctx_type == 'rel_keypoint_splithead_unshared_nonlinear':
        keypoint_linear = NonlinearMappingNetwork(input_dim=rpe_config.kp_dim,
                                                  n_kps=rpe_config.num_keypoints,
                                                  c1=32, c2=num_buckets * num_heads * depth, use_mlp=True)

    else:
        raise ValueError(f'Not support ctx_type: {rpe_config.ctx_type}')

    return keypoint_linear, num_buckets



class NonlinearMappingNetwork(nn.Module):
    def __init__(self, input_dim=512, n_kps=19, c1=32, c2=19200, use_mlp=False, inner_mlp_dim=512):
        super(NonlinearMappingNetwork, self).__init__()
        self.c1 = c1
        self.c2 = c2
        # input shape [8, 148, 19, 512]

        # 1. Map the last dim 512 to c1 (e.g., 32)
        self.linear1 = nn.Linear(input_dim, c1)
        
        # 3. Map the last dim to C2 (e.g., 19200)
        if use_mlp:
            self.fc = nn.Sequential(
                nn.Linear(n_kps * c1, inner_mlp_dim),
                nn.ReLU(),
                nn.Linear(inner_mlp_dim, c2)
            )
            # init zero
            self.fc[-1].weight.data.zero_()
            self.fc[-1].bias.data.zero_()
        else:
            self.fc = nn.Linear(n_kps * c1, c2)
            # init zero
            self.fc[-1].weight.data.zero_()
            self.fc[-1].bias.data.zero_()
    
    def forward(self, x):
        # Input shape: [8, 148, 19, 512]
        x = self.linear1(x)  # Output shape: [8, 148, 19, c1]
        x = x.flatten(2)  # Output shape: [8, 148, 19*c1]
        x = self.fc(x)  # Output shape: [8, 148, c2]
        return x



def make_kprpe_input(rel_kp_embs, keypoint_linear, rpe_config, depth, num_heads, num_buckets):
    B = rel_kp_embs.shape[0]
    # rel_kp_embs: B, n_patches, n_kps, kp_dim
    ctx_type = rpe_config.get('ctx_type', '')
    num_kp = rpe_config.num_keypoints
    if ctx_type == 'rel_keypoint_splithead_unshared':
        rel_keypoints = rel_kp_embs[:, :, :num_kp].flatten(2) # B x n_patches x (n_kps * kp_dim) 
        rel_keypoints = keypoint_linear(rel_keypoints)  # B x n_patches x C where C is num_heads * depth * num_buckets
        rel_keypoints = rel_keypoints.view(B, -1, num_heads * depth, num_buckets).transpose(1, 2)
        rel_keypoints = torch.chunk(rel_keypoints, depth, dim=1)
        extra_ctx = [{'rel_keypoints': rel_keypoint} for rel_keypoint in rel_keypoints]
    elif ctx_type == 'rel_keypoint_splithead_unshared_nonlinear':
        rel_keypoints = rel_kp_embs[:, :, :num_kp]  # B x n_patches x n_kps x kp_dim
        rel_keypoints = keypoint_linear(rel_keypoints)  # B x n_patches x C where C is num_heads * depth * num_buckets
        rel_keypoints = rel_keypoints.view(B, -1, num_heads * depth, num_buckets).transpose(1, 2)
        rel_keypoints = torch.chunk(rel_keypoints, depth, dim=1)
        extra_ctx = [{'rel_keypoints': rel_keypoint} for rel_keypoint in rel_keypoints]
    else:
        raise ValueError(f'Not support ctx_type: {ctx_type}')

    return extra_ctx