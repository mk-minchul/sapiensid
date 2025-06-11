import os
import numpy as np
import torch
import time
try:
    import metric
except:
    from . import metric

import torch.nn.functional as F

def evaluate(
        embeddings,
        image_paths,
        pids,
        camids,
        clothes_ids,
        meta,
        save_artifacts_path='',
):
    
    query_index = [i for i, x in enumerate(meta['full_meta_data_label']) if x == 'query']
    gallery_index = [i for i, x in enumerate(meta['full_meta_data_label']) if x == 'gallery']
    print('Number of query: ', len(query_index))
    print('Number of gallery: ', len(gallery_index))
    assert len(embeddings) == len(query_index) + len(gallery_index)

    qf = embeddings[query_index]
    gf = embeddings[gallery_index]
    q_pids = pids[query_index]
    g_pids = pids[gallery_index]
    q_camids = camids[query_index]
    g_camids = camids[gallery_index]
    q_clothes_ids = clothes_ids[query_index]
    g_clothes_ids = clothes_ids[gallery_index]

    if qf.shape[1] == 3078:
        from torchreid.metrics.distance import compute_distance_matrix_using_bp_features
        # kpr
        q_f, qf_parts_visibility = qf[:, :6*512], qf[:, 6*512:]
        q_f = q_f.view(q_f.size(0), 6, 512)
        g_f, gf_parts_visibility = gf[:, :6*512], gf[:, 6*512:]
        g_f = g_f.view(g_f.size(0), 6, 512)

        qf = F.normalize(q_f, p=2, dim=-1)
        gf = F.normalize(g_f, p=2, dim=-1)
        distmat, body_parts_distmat = compute_distance_matrix_using_bp_features(
            qf,
            gf,
            qf_parts_visibility,
            gf_parts_visibility,
            'mean',
            500,
            True,
            "euclidean",
        )
        distmat = distmat.numpy()
    else:
        # Compute distance matrix between query and gallery
        m, n = qf.size(0), gf.size(0)
        distmat = torch.zeros((m,n))
        qf, gf = qf.cuda(), gf.cuda()
        # Cosine similarity
        for i in range(m):
            distmat[i] = (- torch.mm(qf[i:i+1], gf.t())).cpu()
        distmat = distmat.numpy()

    q_pids, q_camids, q_clothes_ids = q_pids.numpy(), q_camids.numpy(), q_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()

    if save_artifacts_path:
        to_save = {}
        to_save['distmat'] = distmat
        to_save['q_pids'] = q_pids
        to_save['g_pids'] = g_pids
        to_save['q_camids'] = q_camids
        to_save['g_camids'] = g_camids
        to_save['q_clothes_ids'] = q_clothes_ids
        to_save['g_clothes_ids'] = g_clothes_ids
        os.makedirs(os.path.dirname(save_artifacts_path), exist_ok=True)
        torch.save(to_save, save_artifacts_path)

    results = {}

    # cmc, mAP = metric.evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    from .metric_gpu import evaluate as evaluate_gpu
    cmc, mAP = evaluate_gpu(distmat, q_pids, g_pids, q_camids, g_camids)

    results['overall_top1'] = cmc[0]
    results['overall_top5'] = cmc[4]
    results['overall_top10'] = cmc[9]
    results['overall_top20'] = cmc[19]
    results['overall_mAP'] = mAP

    # cmc, mAP = metric.evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='SC')
    from .metric_gpu import evaluate_with_clothes as evaluate_with_clothes_gpu
    cmc, mAP = evaluate_with_clothes_gpu(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='SC')

    results['same_clothes_top1'] = cmc[0]
    results['same_clothes_top5'] = cmc[4]
    results['same_clothes_top10'] = cmc[9]
    results['same_clothes_top20'] = cmc[19]
    results['same_clothes_mAP'] = mAP

    # cmc, mAP = metric.evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='CC')
    from .metric_gpu import evaluate_with_clothes as evaluate_with_clothes_gpu
    cmc, mAP = evaluate_with_clothes_gpu(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='CC')

    results['diff_clothes_top1'] = cmc[0]
    results['diff_clothes_top5'] = cmc[4]
    results['diff_clothes_top10'] = cmc[9]
    results['diff_clothes_top20'] = cmc[19]
    results['diff_clothes_mAP'] = mAP

    # multiply by 100
    for k, v in results.items():
        results[k] = v * 100

    return results