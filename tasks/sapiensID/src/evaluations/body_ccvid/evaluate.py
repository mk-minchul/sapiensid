import os
import numpy as np
import torch
import time
try:
    import metric
except:
    from . import metric

def evaluate(
        embeddings,
        image_paths,
        pids,
        camids,
        clothes_ids,
        meta,
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