import torch
from tqdm import tqdm
from torch.nn import functional as F


def find_too_sim_to_val_center(centers_dict, val_center, threshold, use_cuda=True):
    too_sim_to_val_center = []
    val_center = torch.nn.functional.normalize(val_center, p=2, dim=1)
    device = torch.device('cuda' if use_cuda else 'cpu')
    val_center = val_center.to(device)
    for key, feat in tqdm(centers_dict.items(), desc='find_too_sim_to_val_center', total=len(centers_dict), ncols=75, miniters=10):
        normalized_feat = F.normalize(feat.to(device), p=2, dim=0)
        sim_to_val_center = normalized_feat @ val_center.T
        max_val = sim_to_val_center.max()
        if max_val > threshold:
            # print(f'center {key} has high similarity to val_center: {max_val}')
            too_sim_to_val_center.append(key)

    return too_sim_to_val_center