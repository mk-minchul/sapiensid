import numpy as np
import torch
import math

def compute_ap_cmc(index, good_index, junk_index, device):
    """ Compute AP and CMC for each sample using PyTorch tensors on GPU """
    ap = 0.0
    cmc = torch.zeros(index.size(0), device=device)
    
    # Remove junk_index from index
    if junk_index.numel() > 0:
        mask = ~torch.isin(index, junk_index)
        index = index[mask]
    
    # Find positions of good_index in index
    mask = torch.isin(index, good_index)
    rows_good = torch.nonzero(mask).view(-1)
    
    ngood = good_index.numel()
    if ngood == 0 or rows_good.numel() == 0:
        return ap, cmc

    # Sort rows_good to ensure correct order
    rows_good = rows_good.sort()[0]
    
    cmc[rows_good[0]:] = 1.0

    # Compute Average Precision (AP)
    num_rel = rows_good.numel()
    d_recall = 1.0 / ngood
    precision = (torch.arange(1, num_rel + 1, device=device)) / (rows_good + 1.0)
    ap = torch.sum(d_recall * precision).item()

    return ap, cmc


def compute_ap_cmc_batch(order, good_index, junk_index, device, max_rank=None):
    """Compute AP and CMC for each sample using PyTorch tensors on GPU."""
    if max_rank is None:
        max_rank = order.size(0)
    else:
        max_rank = min(max_rank, order.size(0))

    ap = 0.0
    cmc = torch.zeros(max_rank, device=device)

    # Remove junk_index from order
    if junk_index.numel() > 0:
        mask = ~torch.isin(order, junk_index)
        order = order[mask]

    # Find positions of good_index in order
    mask = torch.isin(order, good_index)
    rows_good = torch.nonzero(mask).view(-1)

    if rows_good.numel() == 0:
        return ap, cmc

    # Compute CMC and AP
    rows_good = rows_good.sort()[0]
    cmc[rows_good[0]:] = 1.0

    ngood = good_index.numel()
    num_rel = rows_good.numel()
    d_recall = 1.0 / ngood
    precision = (torch.arange(1, num_rel + 1, device=device)) / (rows_good + 1.0)
    ap = torch.sum(d_recall * precision).item()

    return ap, cmc



def compute_far_threshold(distmat, q_pids, g_pids, far_target=0.01):
    """
    Compute the similarity threshold at a given FAR.

    Args:
        distmat (torch.Tensor): Distance matrix of shape (num_queries, num_gallery).
        q_pids (torch.Tensor): Query person IDs of shape (num_queries,).
        g_pids (torch.Tensor): Gallery person IDs of shape (num_gallery,).
        far_target (float): Target false acceptance rate (e.g., 0.01 for FAR@0.01).

    Returns:
        float: Similarity threshold at the specified FAR.
    """
    assert distmat.dim() == 2, "distmat must be a 2D tensor"
    num_q, num_g = distmat.shape
    
    device = distmat.device

    # Flatten the distance matrix and create corresponding labels
    # 1 indicates "genuine" (same ID); 0 indicates "impostor" (different ID)
    distances = distmat.flatten()
    q_pids_repeated = q_pids.unsqueeze(1).expand(num_q, num_g).flatten()
    g_pids_repeated = g_pids.unsqueeze(0).expand(num_q, num_g).flatten()

    labels = (q_pids_repeated == g_pids_repeated).int()

    # Filter impostor distances
    impostor_distances = distances[labels == 0]

    # Sort impostor distances
    impostor_distances = torch.sort(impostor_distances)[0]

    # Compute the threshold index for the given FAR target
    threshold_index = math.ceil(far_target * len(impostor_distances)) - 1
    threshold_index = max(0, threshold_index)  # Ensure the index is non-negative

    # Get the threshold distance
    threshold = impostor_distances[threshold_index].item()

    return threshold


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, ignore_cam=False, batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_q, num_g = distmat.shape

    # Convert numpy arrays to PyTorch tensors and move to GPU
    if isinstance(distmat, np.ndarray):
        distmat = torch.from_numpy(distmat)
    if isinstance(q_pids, np.ndarray):
        q_pids = torch.from_numpy(q_pids).to(device)
    else:
        q_pids = q_pids.to(device)
    if isinstance(g_pids, np.ndarray):
        g_pids = torch.from_numpy(g_pids).to(device)
    else:
        g_pids = g_pids.to(device)
    if isinstance(q_camids, np.ndarray):
        q_camids = torch.from_numpy(q_camids).to(device)
    else:
        q_camids = q_camids.to(device)
    if isinstance(g_camids, np.ndarray):
        g_camids = torch.from_numpy(g_camids).to(device)
    else:
        g_camids = g_camids.to(device)
    
    if (g_camids == -1).all():
        ignore_cam = True

    num_no_gt = 0  # Number of queries without valid ground truth
    CMC = torch.zeros(num_g, device=device)
    AP = 0.0

    # Process queries in batches
    num_batches = (num_q + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_q)
        batch_distmat = distmat[start_idx:end_idx].to(device)
        batch_q_pids = q_pids[start_idx:end_idx].to(device)
        batch_q_camids = q_camids[start_idx:end_idx].to(device)

        # Compute indices of sorted distances for the batch
        _, indices = torch.sort(batch_distmat, dim=1)

        for i in range(indices.size(0)):
            # Index in the original dataset
            idx = start_idx + i
            q_pid = batch_q_pids[i]
            q_camid = batch_q_camids[i]

            # Create masks for gallery samples
            pid_mask = (g_pids == q_pid)
            camid_mask = (g_camids == q_camid)

            # Determine good and junk indices based on the evaluation protocol
            if ignore_cam:
                # Include same-camera matches
                good_index = torch.nonzero(pid_mask).view(-1)
                junk_index = torch.tensor([], dtype=torch.long, device=device)
            else:
                # Exclude same-camera matches
                good_index = torch.nonzero(pid_mask & (~camid_mask)).view(-1)
                junk_index = torch.nonzero(pid_mask & camid_mask).view(-1)

            # Remove invalid indices (e.g., query sample itself in gallery)
            if not ignore_cam:
                invalid_index = torch.nonzero((q_pid == g_pids) & (q_camid == g_camids)).view(-1)
                junk_index = torch.cat((junk_index, invalid_index))

            if good_index.numel() == 0:
                num_no_gt += 1
                continue

            # Get sorted indices for current query
            order = indices[i]

            ap_tmp, cmc_tmp = compute_ap_cmc(order, good_index, junk_index, device)

            CMC += cmc_tmp
            AP += ap_tmp
        

    if num_no_gt > 0:
        print(f"{num_no_gt} query samples do not have valid ground truth.")

    valid_queries = num_q - num_no_gt
    if valid_queries == 0:
        raise RuntimeError("No valid query samples were found.")

    CMC = CMC / valid_queries
    mAP = AP / valid_queries

    # Move results back to CPU and convert to numpy
    CMC = CMC.cpu().numpy()
    mAP = float(mAP)

    thresholds = np.arange(-1, 1, 0.01)
    # Reshape q_pids and g_pids for broadcasting
    q_pids_exp = q_pids.view(-1, 1)  # Shape [493, 1]
    g_pids_exp = g_pids.view(1, -1)  # Shape [1, 7050]

    # Positive and negative masks
    pos_mask = (q_pids_exp == g_pids_exp)  # Positive pair mask [493, 7050]
    neg_mask = ~pos_mask  # Negative pair mask [493, 7050]

    # Extract positive and negative pair distances
    pos_pair_dist = distmat[pos_mask.cpu()]
    neg_pair_dist = distmat[neg_mask.cpu()][:len(pos_pair_dist)]
    actual_issame = torch.cat([torch.ones(len(pos_pair_dist)), torch.zeros(len(neg_pair_dist))])

    # Combine distances
    combined_dist = torch.cat([pos_pair_dist, neg_pair_dist])

    from sklearn.model_selection import KFold
    class LFold:
        def __init__(self, n_splits=2, shuffle=False):
            self.n_splits = n_splits
            if self.n_splits > 1:
                self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

        def split(self, indices):
            if self.n_splits > 1:
                return self.k_fold.split(indices)
            else:
                return [(indices, indices)]

    def calculate_accuracy(threshold, dist, actual_issame):
        predict_issame = np.less(dist, threshold)
        tp = torch.sum(torch.logical_and(predict_issame, actual_issame))
        fp = torch.sum(torch.logical_and(predict_issame, torch.logical_not(actual_issame)))
        tn = torch.sum(
            torch.logical_and(torch.logical_not(predict_issame),
                        torch.logical_not(actual_issame)))
        fn = torch.sum(torch.logical_and(torch.logical_not(predict_issame), actual_issame))

        tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
        fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
        acc = float(tp + tn) / len(dist)
        return tpr, fpr, acc

    nrof_folds=10
    nrof_pairs = len(combined_dist)
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    best_thresholds = []
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, combined_dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, combined_dist[test_set],
                actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], combined_dist[test_set],
            actual_issame[test_set])
        best_thresholds.append(thresholds[best_threshold_index])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)

    print(f"Best thresholds: {best_thresholds}")


    return CMC, mAP



def evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids,
                          q_clothids, g_clothids, mode='CC', batch_size=128, max_rank=50):
    """Compute CMC and mAP with clothes using PyTorch tensors on GPU, processing queries in batches.

    Args:
        distmat (numpy ndarray): Distance matrix with shape (num_query, num_gallery).
        q_pids (numpy array): Person IDs for query samples.
        g_pids (numpy array): Person IDs for gallery samples.
        q_camids (numpy array): Camera IDs for query samples.
        g_camids (numpy array): Camera IDs for gallery samples.
        q_clothids (numpy array): Clothes IDs for query samples.
        g_clothids (numpy array): Clothes IDs for gallery samples.
        mode (str): 'CC' for clothes-changing; 'SC' for the same clothes.
        batch_size (int): Number of queries to process in a batch.
        max_rank (int): Maximum rank to consider for CMC computation.
    """
    assert mode in ['CC', 'SC']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_q, num_g = distmat.shape
    max_rank = min(max_rank, num_g)

    # Convert numpy arrays to PyTorch tensors
    if isinstance(distmat, np.ndarray):
        distmat = torch.from_numpy(distmat)
    if isinstance(q_pids, np.ndarray):
        q_pids = torch.from_numpy(q_pids)
    else:
        q_pids = q_pids.to(device)
    if isinstance(g_pids, np.ndarray):
        g_pids = torch.from_numpy(g_pids).to(device)
    else:
        g_pids = g_pids.to(device)
    if isinstance(q_camids, np.ndarray):
        q_camids = torch.from_numpy(q_camids).to(device)
    else:
        q_camids = q_camids.to(device)
    if isinstance(g_camids, np.ndarray):
        g_camids = torch.from_numpy(g_camids).to(device)
    else:
        g_camids = g_camids.to(device)
    if isinstance(q_clothids, np.ndarray):
        q_clothids = torch.from_numpy(q_clothids).to(device)
    else:
        q_clothids = q_clothids.to(device)
    if isinstance(g_clothids, np.ndarray):
        g_clothids = torch.from_numpy(g_clothids).to(device)
    else:
        g_clothids = g_clothids.to(device)

    num_no_gt = 0  # Number of queries without valid ground truth
    CMC = torch.zeros(max_rank)
    AP = 0.0

    # Process queries in batches
    num_batches = (num_q + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_q)
        batch_distmat = distmat[start_idx:end_idx].to(device)
        batch_q_pids = q_pids[start_idx:end_idx].to(device)
        batch_q_camids = q_camids[start_idx:end_idx].to(device)
        batch_q_clothids = q_clothids[start_idx:end_idx].to(device)

        # Compute indices of sorted distances for the batch
        _, indices = torch.sort(batch_distmat, dim=1)

        for i in range(indices.size(0)):
            idx = start_idx + i
            q_pid = batch_q_pids[i]
            q_camid = batch_q_camids[i]
            q_clothid = batch_q_clothids[i]

            # Create masks for gallery samples
            pid_mask = (g_pids == q_pid)
            camid_mask = (g_camids == q_camid)
            clothid_mask = (g_clothids == q_clothid)

            if mode == 'CC':
                # Clothes-changing mode
                # Good indices: same PID, different CAMID, different cloth ID
                good_mask = pid_mask & (~camid_mask) & (~clothid_mask)
                good_index = torch.nonzero(good_mask).view(-1)

                # Junk indices: same PID and same CAMID, or same PID and same cloth ID
                junk_mask1 = pid_mask & camid_mask
                junk_mask2 = pid_mask & clothid_mask
                junk_index1 = torch.nonzero(junk_mask1).view(-1)
                junk_index2 = torch.nonzero(junk_mask2).view(-1)
                junk_index = torch.cat((junk_index1, junk_index2)).unique()

            elif mode == 'SC':
                # Same-clothes mode
                # Good indices: same PID, different CAMID, same cloth ID
                good_mask = pid_mask & (~camid_mask) & clothid_mask
                good_index = torch.nonzero(good_mask).view(-1)

                # Junk indices: same PID and same CAMID, or same PID and different cloth ID
                junk_mask1 = pid_mask & camid_mask
                junk_mask2 = pid_mask & (~clothid_mask)
                junk_index1 = torch.nonzero(junk_mask1).view(-1)
                junk_index2 = torch.nonzero(junk_mask2).view(-1)
                junk_index = torch.cat((junk_index1, junk_index2)).unique()

            else:
                raise ValueError("Invalid mode. Mode should be 'CC' or 'SC'.")

            if good_index.numel() == 0:
                num_no_gt += 1
                continue

            # Get sorted indices for current query
            order = indices[i]

            # Compute AP and CMC
            ap_tmp, cmc_tmp = compute_ap_cmc_batch(order, good_index, junk_index, device, max_rank)

            CMC += cmc_tmp.cpu()
            AP += ap_tmp

        # Clear variables to free up GPU memory
        del batch_distmat, batch_q_pids, batch_q_camids, batch_q_clothids, indices
        torch.cuda.empty_cache()

    if num_no_gt > 0:
        print(f"{num_no_gt} query samples do not have valid ground truth.")

    valid_queries = num_q - num_no_gt
    if valid_queries == 0:
        raise RuntimeError("No valid query samples were found.")

    CMC = CMC / valid_queries
    mAP = AP / valid_queries

    # Convert results to numpy
    CMC = CMC.numpy()
    mAP = float(mAP)

    return CMC, mAP

if __name__ == '__main__':

    np.random.seed(0)
    distmat = np.random.rand(1000, 10000)
    q_pids = np.random.randint(0, 100, size=(1000,))
    g_pids = np.random.randint(0, 100, size=(10000,))
    q_camids = np.random.randint(0, 10, size=(1000,))
    g_camids = np.random.randint(0, 10, size=(10000,))

    # Evaluate
    CMC, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print(f"mAP: {mAP}")
    print(f"CMC curve: {CMC}")

    # Query samples: images of persons from camera 0
    q_pids = np.array([0, 1, 2, 3, 4])
    q_camids = np.array([0, 1, 2, 3, 4])

    # Gallery samples: images of persons from camera 1
    g_pids = np.array([0, 1, 2, 3, 4])
    g_camids = np.array([0, 1, 2, 3, 4])+6

    # Construct a distance matrix where the distance between the same person is small,
    # and the distance between different persons is large
    # For simplicity, we'll set the distance to 0.1 for matching pairs and 0.9 for non-matching pairs
    num_q = len(q_pids)
    num_g = len(g_pids)
    distmat = np.full((num_q, num_g), 0.9)

    # Set small distances for matching person IDs
    for i in range(num_q):
        distmat[i, i] = 0.1  # Matching pair has a small distance

    # Now, let's call the evaluate function with this test data
    CMC, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print(f"mAP: {mAP}")
    print(f"CMC curve: {CMC}")
