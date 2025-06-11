import torch
from torch import nn
from functools import partial
from .multihead_diffattn import MultiheadDiffAttn
import pandas as pd
import os


class SemanticAttentionHead(nn.Module):
    def __init__(self, head_config):
        super().__init__()
        self.head_config = head_config

        self.n_multiply_per_query = self.head_config.n_multiply_per_query
        self.n_query = self.head_config.num_keypoints*self.head_config.n_multiply_per_query
        self.query_pos_emb = nn.Parameter(torch.randn(1, self.n_query, self.head_config.emb_dim))
        nn.init.normal_(self.query_pos_emb, std=0.02)

        self.attn = MultiheadDiffAttn(
            embed_dim=self.head_config.emb_dim,
            depth=self.head_config.depth,
            num_heads=self.head_config.num_heads,
            model_parallel_size=self.head_config.model_parallel_size,
            decoder_kv_attention_heads=self.head_config.decoder_kv_attention_heads
        )

        self.nparts = self.head_config.num_keypoints * 2 * self.head_config.n_multiply_per_query 
        self.flatten_feature = nn.Sequential(
            nn.Linear(in_features=self.head_config.emb_dim * self.nparts, 
                      out_features=self.head_config.output_dim, bias=False),
            nn.BatchNorm1d(num_features=self.head_config.output_dim, eps=2e-5),
            nn.Linear(in_features=self.head_config.output_dim, out_features=self.head_config.output_dim, bias=False),
            nn.BatchNorm1d(num_features=self.head_config.output_dim, eps=2e-5)
        )


        self.n_dataset = 1  # set it to more than 1 if you want to use multiple datasets
        # record_reader_indices is the index of the dataset for each sample.
        # record_reader_indices must be between 0 and self.n_dataset - 1 during training
        self.learned_mask = nn.Parameter(torch.randn(self.n_dataset, self.nparts))
        self.learned_mask.data.fill_(0)


    def forward(self, x, x_mask_ids_keep, pos_embs, keypoint_embs, return_intermediate=False, ldmks=None, record_reader_indices=None, *args, **kwargs):

        # some keypoint_embs could be all zero if it is missing keypoint
        if x.size(1) - 1 == pos_embs.size(1):
            x = x[:, 1:]

        keypoint_embs = keypoint_embs.transpose(1, 2).unsqueeze(2)
        keypoint_embs = keypoint_embs.repeat(1, 1, self.n_multiply_per_query, 1)
        keypoint_embs = keypoint_embs.flatten(1, 2)
        keypoint_embs = keypoint_embs + self.query_pos_emb

        # learnable keypoint pooling
        x_learn_pooled, attn_weights = self.attn(query=keypoint_embs, key=pos_embs, value=x)

        is_missing_ldmk = (ldmks == -1).all(2).unsqueeze(2).repeat(1, 1, self.n_multiply_per_query).flatten(1, 2)
        x_learn_pooled[is_missing_ldmk] = 0
        
        # pool by keypoint's most similar position
        with torch.no_grad():
            norm_fn = partial(torch.nn.functional.normalize, p=2)
            sim = norm_fn(pos_embs, dim=2) @ norm_fn(keypoint_embs.transpose(1, 2), dim=1)
            sim = sim.detach()
        topk_sim, topk_idx = sim.topk(k=1, dim=1)
        topk_idx = topk_idx.transpose(1, 2).flatten(1).unsqueeze(2)
        x_point_selected = torch.gather(x, 1, topk_idx.expand(-1, -1, x.size(2)))
        x_point_selected = x_point_selected.view(x.size(0), -1, x.size(2))
        x_point_selected[is_missing_ldmk] = 0

        features = torch.cat([x_learn_pooled, x_point_selected], dim=1)

        if self.training and record_reader_indices is not None:
            data_selector = torch.nn.functional.one_hot(record_reader_indices, self.n_dataset).float()
            mask = torch.matmul(data_selector, self.learned_mask.sigmoid())
        else:
            mask_idx = 0
            mask = self.learned_mask.sigmoid()[mask_idx].unsqueeze(0).repeat(x.size(0), 1)
        features = features * mask.unsqueeze(2)

        flat_features = features.flatten(1, 2)
        out = self.flatten_feature(flat_features)

        is_missing_ldmk_mask = torch.cat([is_missing_ldmk, is_missing_ldmk], dim=1)

        if return_intermediate:
            return out, (features, is_missing_ldmk_mask)
        return out


