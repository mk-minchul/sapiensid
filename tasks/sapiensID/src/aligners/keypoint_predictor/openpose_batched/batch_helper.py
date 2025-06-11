import numpy as np
import torch
import math
import torch.nn.functional as F

def smart_resize_k_batch(x, fx, fy):
    # x: B x C x H x W
    # Output: resized x of shape B x C x H', W', where H' = H * fy, W' = W * fx
    B, C, H, W = x.shape
    Ht = int(H * fy)
    Wt = int(W * fx)
    x_resized = F.interpolate(x, size=(Ht, Wt), mode='bilinear', align_corners=False)
    return x_resized

def padRightDownCorner_batch(imgs, stride, padValue):
    # imgs: B x C x H x W
    # Output: imgs_padded: B x C x H', W', pad: [pad_up, pad_left, pad_down, pad_right]
    B, C, H, W = imgs.shape

    pad_up = 0
    pad_left = 0
    pad_down = 0 if H % stride == 0 else stride - (H % stride)
    pad_right = 0 if W % stride == 0 else stride - (W % stride)

    padding = (pad_left, pad_right, pad_up, pad_down)  # left, right, top, bottom
    imgs_padded = F.pad(imgs, padding, mode='constant', value=padValue)

    pad = [pad_up, pad_left, pad_down, pad_right]
    return imgs_padded, pad

def get_gaussian_kernel2d(kernel_size, sigma):
    """Creates a 2D Gaussian kernel."""
    x_coord = torch.arange(kernel_size).float()
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance)
    )

    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    return gaussian_kernel


def batched_find_peak(heatmap_avg, paf_avg, thre1):
    """
    Performs batched peak finding on heatmaps.

    Args:
        heatmap_avg: Tensor of shape (B, 19, H, W)
        paf_avg: Tensor of shape (B, 38, H, W)
        thre1: Threshold value for peaks

    Returns:
        all_peaks: List of length B, where each element is a list of peaks per part.
                   Each element is a list of length 18 (for 18 parts), and each part contains a list of peaks.
    """
    device = heatmap_avg.device
    B, _, H, W = heatmap_avg.shape

    # Select heatmaps for parts
    heatmaps = heatmap_avg[:, :18, :, :]  # shape (B, 18, H, W)

    # Apply Gaussian filter
    sigma = 3
    kernel_size = int(6 * sigma + 1)

    gaussian_kernel = get_gaussian_kernel2d(kernel_size=kernel_size, sigma=sigma).to(device)  # shape (kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)

    padding = kernel_size // 2

    heatmaps_reshaped = heatmaps.reshape(-1, 1, H, W)  # shape (B*18, 1, H, W)

    smoothed_heatmaps = F.conv2d(heatmaps_reshaped, gaussian_kernel, padding=padding)
    smoothed_heatmaps = smoothed_heatmaps.view(B, 18, H, W)  # shape (B, 18, H, W)

    # Create shifted versions
    map_left = torch.zeros_like(smoothed_heatmaps)
    map_left[:, :, 1:, :] = smoothed_heatmaps[:, :, :-1, :]

    map_right = torch.zeros_like(smoothed_heatmaps)
    map_right[:, :, :-1, :] = smoothed_heatmaps[:, :, 1:, :]

    map_up = torch.zeros_like(smoothed_heatmaps)
    map_up[:, :, :, 1:] = smoothed_heatmaps[:, :, :, :-1]

    map_down = torch.zeros_like(smoothed_heatmaps)
    map_down[:, :, :, :-1] = smoothed_heatmaps[:, :, :, 1:]

    # Compute peaks_binary
    peaks_binary = (smoothed_heatmaps >= map_left) & \
                   (smoothed_heatmaps >= map_right) & \
                   (smoothed_heatmaps >= map_up) & \
                   (smoothed_heatmaps >= map_down) & \
                   (smoothed_heatmaps > thre1)

    # Get indices of peaks
    indices = torch.nonzero(peaks_binary, as_tuple=False)  # shape (N, 4)

    # Get scores from original heatmaps
    map_ori = heatmaps  # shape (B, 18, H, W)
    scores = map_ori[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]]  # shape (N,)

    # Initialize list to store peaks for each batch
    all_peaks = []

    # Process peaks per batch item
    for b in range(B):
        indices_b = indices[indices[:, 0] == b]
        scores_b = scores[indices[:, 0] == b]

        all_peaks_b = []
        peak_counter_b = 0  # counts for this batch item

        for part in range(18):
            indices_bp = indices_b[indices_b[:, 1] == part]
            scores_bp = scores_b[indices_b[:, 1] == part]

            peaks = indices_bp[:, 2:]  # positions h, w

            # Convert to x, y coordinates (w, h)
            peaks_xy = peaks[:, [1, 0]].cpu().numpy()

            # Get scores
            peaks_scores = scores_bp.cpu().numpy()

            # Create peaks_with_score
            peaks_with_score = [tuple(peaks_xy[i]) + (peaks_scores[i],) for i in range(len(peaks_scores))]

            # Assign peak IDs
            peak_ids = list(range(peak_counter_b, peak_counter_b + len(peaks_with_score)))

            peaks_with_score_and_id = [peaks_with_score[i] + (peak_ids[i],) for i in range(len(peak_ids))]

            all_peaks_b.append(peaks_with_score_and_id)

            peak_counter_b += len(peaks_with_score)

        all_peaks.append(all_peaks_b)

    return all_peaks



def body_call_batch(model, input_images):
    # input_images: B x 3 x H x W, values between 0 and 1
    # Returns: candidates, subsets for each image in the batch
    device = next(iter(model.parameters())).device
    scale_search = [0.5]
    boxsize = 368
    stride = 8
    padValue = 128
    thre1 = 0.1
    thre2 = 0.05

    B, C, H_ori, W_ori = input_images.shape

    multiplier = [x * boxsize / H_ori for x in scale_search]

    # Initialize heatmap_avg and paf_avg
    heatmap_avg = torch.zeros((B, 19, H_ori, W_ori), device=device)
    paf_avg = torch.zeros((B, 38, H_ori, W_ori), device=device)

    for scale in multiplier:
        # Resize the images
        input_images_resized = smart_resize_k_batch(input_images, fx=scale, fy=scale)  # B x 3 x H', W'

        # Pad images
        input_images_padded, pad = padRightDownCorner_batch(input_images_resized, stride, padValue)

        # Prepare the data tensor for input to the model
        data = input_images_padded.float() - 0.5  # B x 3 x H_padded x W_padded
        data = data.to(device)

        # Run the model
        with torch.no_grad():
            Mconv7_stage6_L1, Mconv7_stage6_L2 = model(data)

        # Upsample the outputs to the original scale
        heatmap = F.interpolate(Mconv7_stage6_L2, scale_factor=stride, mode='bilinear', align_corners=False)
        paf = F.interpolate(Mconv7_stage6_L1, scale_factor=stride, mode='bilinear', align_corners=False)

        # Remove padding
        pad_down = pad[2]
        pad_right = pad[3]
        H_padded = input_images_padded.shape[2]
        W_padded = input_images_padded.shape[3]
        H = H_padded - pad_down
        W = W_padded - pad_right

        heatmap = heatmap[:, :, :H, :W]
        paf = paf[:, :, :H, :W]

        # Resize to original image size
        heatmap = F.interpolate(heatmap, size=(H_ori, W_ori), mode='bilinear', align_corners=False)
        paf = F.interpolate(paf, size=(H_ori, W_ori), mode='bilinear', align_corners=False)

        # Aggregate
        heatmap_avg += heatmap / len(multiplier)
        paf_avg += paf / len(multiplier)

    # Now process heatmap_avg and paf_avg to get the keypoints
    candidates = []
    subsets = []

    batched_all_peaks = batched_find_peak(heatmap_avg, paf_avg, thre1)
    heatmap_avg_np = heatmap_avg.cpu().numpy()
    paf_avg_np = paf_avg.cpu().numpy()

    for b in range(B):
        all_peaks = batched_all_peaks[b]
        heatmap_avg_b = heatmap_avg_np[b].transpose(1, 2, 0)  # H x W x 19
        paf_avg_b = paf_avg_np[b].transpose(1, 2, 0)  # H x W x 38

        # find connection in the specified sequence
        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9],
                   [9, 10], [10, 11], [2, 12], [12, 13], [13, 14], [2, 1],
                   [1, 15], [15, 17], [1, 16], [16, 18], [3, 17], [6, 18]]
        mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44],
                  [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30],
                  [47, 48], [49, 50], [53, 54], [51, 52], [55, 56],
                  [37, 38], [45, 46]]

        connection_all = []
        special_k = []
        mid_num = 10

        for k in range(len(mapIdx)):
            score_mid = paf_avg_b[:, :, [x - 19 for x in mapIdx[k]]]
            candA = all_peaks[limbSeq[k][0] - 1]
            candB = all_peaks[limbSeq[k][1] - 1]
            nA = len(candA)
            nB = len(candB)
            indexA, indexB = limbSeq[k]
            if (nA != 0 and nB != 0):
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        norm = np.linalg.norm(vec)
                        if norm == 0:
                            continue
                        vec = vec / norm

                        startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                                            np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                        vec_x = np.array([score_mid[int(round(startend[I][1])),
                                                    int(round(startend[I][0])), 0] for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])),
                                                    int(round(startend[I][0])), 1] for I in range(len(startend))])

                        score_midpts = vec_x * vec[0] + vec_y * vec[1]
                        score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                            0.5 * heatmap_avg_b.shape[0] / norm - 1, 0)
                        criterion1 = len(np.nonzero(score_midpts > thre2)[0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append(
                                [i, j, score_with_dist_prior,
                                 score_with_dist_prior + candA[i][2] + candB[j][2]])

                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if (i not in connection[:, 3] and j not in connection[:, 4]):
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        if (len(connection) >= min(nA, nB)):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        # subset: people
        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])

        for k in range(len(mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = np.array(limbSeq[k]) - 1

                for i in range(len(connection_all[k])):
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if subset[j][indexB] != partBs[i]:
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif not found:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])

        # Delete some rows of subset which has few parts
        deleteIdx = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)

        candidates.append(candidate)
        subsets.append(subset)

    return candidates, subsets
