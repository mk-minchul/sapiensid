import os
import numpy as np
import torch
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
        save_artifacts_path,
):
    
    query_same_index = [i for i, x in enumerate(meta['full_meta_data_label']) if x == 'query_same']
    query_diff_index = [i for i, x in enumerate(meta['full_meta_data_label']) if x == 'query_diff']
    gallery_index = [i for i, x in enumerate(meta['full_meta_data_label']) if x == 'gallery']
    print('Number of query same: ', len(query_same_index))
    print('Number of query diff: ', len(query_diff_index))
    print('Number of gallery: ', len(gallery_index))
    assert len(embeddings) == len(query_same_index) + len(query_diff_index) + len(gallery_index)

    qsf = embeddings[query_same_index]
    qdf = embeddings[query_diff_index]
    gf = embeddings[gallery_index]
    qs_pids = pids[query_same_index]
    qd_pids = pids[query_diff_index]
    g_pids = pids[gallery_index]
    qs_camids = camids[query_same_index]
    qd_camids = camids[query_diff_index]
    g_camids = camids[gallery_index]
    qs_clothes_ids = clothes_ids[query_same_index]
    qd_clothes_ids = clothes_ids[query_diff_index]
    g_clothes_ids = clothes_ids[gallery_index]

    m, n, k = qsf.size(0), qdf.size(0), gf.size(0)
    distmat_same = torch.zeros((m, k))
    distmat_diff = torch.zeros((n, k))
    qsf, qdf, gf = qsf.cuda(), qdf.cuda(), gf.cuda()
    # Cosine similarity
    for i in range(m):
        distmat_same[i] = (- torch.mm(qsf[i:i+1], gf.t())).cpu()
    for i in range(n):
        distmat_diff[i] = (- torch.mm(qdf[i:i+1], gf.t())).cpu()
    distmat_same = distmat_same.numpy()
    distmat_diff = distmat_diff.numpy()

    if save_artifacts_path:
        to_save = {}
        to_save['distmat_same'] = distmat_same
        to_save['q_pids'] = qs_pids
        to_save['g_pids'] = g_pids
        to_save['q_camids'] = qs_camids
        to_save['g_camids'] = g_camids
        to_save['distmat_diff'] = distmat_diff
        to_save['qd_pids'] = qd_pids
        to_save['g_pids'] = g_pids
        to_save['qd_camids'] = qd_camids
        to_save['g_camids'] = g_camids
        os.makedirs(os.path.dirname(save_artifacts_path), exist_ok=True)
        torch.save(to_save, save_artifacts_path)

    results = {}

    # cmc, mAP = metric.evaluate(distmat_same, qs_pids, g_pids, qs_camids, g_camids)
    from .metric_gpu import evaluate as evaluate_gpu
    cmc, mAP = evaluate_gpu(distmat_same, qs_pids, g_pids, qs_camids, g_camids)
    results['same_clothes_top1'] = cmc[0]
    results['same_clothes_top5'] = cmc[4]
    results['same_clothes_top10'] = cmc[9]
    results['same_clothes_top20'] = cmc[19]
    results['same_clothes_mAP'] = mAP

    # cmc, mAP = metric.evaluate(distmat_diff, qd_pids, g_pids, qd_camids, g_camids)
    cmc, mAP = evaluate_gpu(distmat_diff, qd_pids, g_pids, qd_camids, g_camids)
    results['diff_clothes_top1'] = cmc[0]
    results['diff_clothes_top5'] = cmc[4]
    results['diff_clothes_top10'] = cmc[9]
    results['diff_clothes_top20'] = cmc[19]
    results['diff_clothes_mAP'] = mAP

    # multiply by 100
    for k, v in results.items():
        results[k] = v * 100
        
    return results