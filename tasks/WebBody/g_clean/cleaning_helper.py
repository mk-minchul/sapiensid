import torch
import torch.nn.functional as F
from tqdm import tqdm
from itertools import combinations
from similarity_helper import compute_similarity


def find_similar_pairs(features_dict, count_dict, batch_size, similarity_threshold, ambiguous_threshold):
    labels = list(features_dict.keys())
    features = torch.stack([features_dict[label] for label in labels])  # Shape: [13,000,000 x 512]
    counts = torch.tensor([count_dict[label] for label in labels])
    labels = torch.tensor(labels)
    similar_pairs, ambiguous_pairs = compute_similarity(features, labels, counts, batch_size, similarity_threshold, ambiguous_threshold)
    return similar_pairs, ambiguous_pairs


def find_ambiguous_pairs(features_dict, count_dict, batch_size, threshold_lower, threshold_upper, use_cuda):

    labels = list(features_dict.keys())
    features = torch.stack([features_dict[label] for label in labels])  # Shape: [13,000,000 x 512]
    counts = torch.tensor([count_dict[label] for label in labels])
    labels = torch.tensor(labels)
    num_samples = features.shape[0]
    ambiguous_pairs = []
    device = 'cuda' if use_cuda else 'cpu'

    for i in tqdm(range(0, num_samples, batch_size), desc='Finding ambiguous pairs', ncols=75, miniters=10, total=num_samples // batch_size):
        # Load a batch of features into GPU
        batch_features = features[i:i+batch_size].to(device)
        batch_labels = labels[i:i+batch_size].to(device)
        batch_counts = counts[i:i+batch_size].to(device)

        # Normalize the batch features
        batch_features = F.normalize(batch_features, p=2, dim=1)

        for j in range(0, num_samples, batch_size):
            # Load another batch of features into GPU
            compare_features = features[j:j+batch_size].to(device)
            compare_labels = labels[j:j+batch_size].to(device)
            compare_counts = counts[j:j+batch_size].to(device)

            # Normalize the compare features
            compare_features = F.normalize(compare_features, p=2, dim=1)

            # Compute cosine similarity
            similarity = torch.mm(batch_features, compare_features.t())

            # Find pairs above threshold
            condition = (similarity > threshold_lower) & (similarity < threshold_upper)

            # Get indices of similar pairs
            row_indices, col_indices = torch.nonzero(condition, as_tuple=True)
            
            # Compute global indices
            global_row_indices = i + row_indices
            global_col_indices = j + col_indices

            # Create a mask for valid pairs (avoid self-comparisons, duplicates, and same labels)
            valid_pairs_mask = (global_row_indices < global_col_indices) & (batch_labels[row_indices] != compare_labels[col_indices])

            # Apply the mask and add to ambiguous_pairs
            label_pairs = torch.stack([
                batch_labels[row_indices[valid_pairs_mask]],
                compare_labels[col_indices[valid_pairs_mask]]
            ], dim=1)
            counts_pairs = torch.stack([
                batch_counts[row_indices[valid_pairs_mask]],
                compare_counts[col_indices[valid_pairs_mask]]
            ], dim=1)

            # sort such that the smaller count is first between each pair
            sorted_indices = torch.argsort(counts_pairs, dim=1)
            sorted_label_pairs = torch.gather(label_pairs, 1, sorted_indices)

            ambiguous_pairs.extend(sorted_label_pairs.tolist())

        # Clear GPU memory
        torch.cuda.empty_cache()
    to_remove_labels = [pair[0] for pair in ambiguous_pairs]
    return to_remove_labels, ambiguous_pairs


def merge_(merged_pairs, centers_dict, count_dict, label_mapping):
    print(f'Merging {len(merged_pairs)} pairs')
    # Updating centers_dict based on merged_pairs
    for pair in merged_pairs:
        key1, key2 = pair
        if key1 not in centers_dict or key2 not in centers_dict:
            continue

        # Retrieve tensors corresponding to the keys
        tensor1 = centers_dict[key1]
        tensor2 = centers_dict[key2]
        count1 = count_dict[key1]
        count2 = count_dict[key2]

        # Calculate the average of the two tensors
        added_tensor = tensor1 + tensor2

        # Update key1 with the averaged tensor
        centers_dict[key1] = added_tensor
        count_dict[key1] = count1 + count2

        # Optionally, remove key2 from the dictionary
        del centers_dict[key2]
        del count_dict[key2]
        label_mapping[key2] = key1
    return centers_dict, count_dict, label_mapping


def delete_(to_remove_labels, centers_dict, count_dict, label_mapping):
    print(f'Removing {len(to_remove_labels)} labels')
    for label in to_remove_labels:
        if label not in centers_dict:
            continue
        del centers_dict[label]
        del count_dict[label]
        label_mapping[label] = -1
    return centers_dict, count_dict, label_mapping


def find_duplicate_images(cluster_df, feature_dataset, similarity_threshold=0.9, use_cuda=True):
    duplicate_pairs = []
    device = 'cuda' if use_cuda else 'cpu'
    
    for label, group in tqdm(cluster_df.groupby('remapped_label'), desc='Finding duplicates', ncols=75):
        image_paths = group['global_path'].values
        if label == -1:
            continue
        if len(image_paths) <= 1:
            continue
        
        features = torch.stack([feature_dataset.read_by_path(img_path)[0] for img_path in image_paths])
        if len(features) > 4096:
            print('somethhing is wrong')
            continue

        features = features.to(device)
        # Normalize features for cosine similarity
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        similarity = torch.mm(features, features.t())

        # Find pairs above threshold
        above_threshold = similarity > similarity_threshold

        # Get indices of similar pairs
        row_indices, col_indices = torch.nonzero(above_threshold, as_tuple=True)
            
        # Create a mask for valid pairs (avoid self-comparisons, duplicates, and same labels)
        valid_pairs_mask = (row_indices < col_indices)
            # Apply the mask and add to similar_pairs
        pairs_indices = torch.stack([
            row_indices[valid_pairs_mask],
            col_indices[valid_pairs_mask]
        ], dim=1).cpu().numpy()
        pairs_paths = []
        image_paths_0 = image_paths[pairs_indices[:, 0]]
        image_paths_1 = image_paths[pairs_indices[:, 1]]
        pairs_paths = list(zip(image_paths_0, image_paths_1))
        duplicate_pairs.extend(pairs_paths)
    
    return duplicate_pairs
