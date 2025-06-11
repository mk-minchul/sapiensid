import torch
import os
from tqdm import tqdm
import networkx as nx

def merge_with_precluster_(similar_components, centers_dict, count_dict, label_mapping, inner_threshold=0.7):
    print(f'Merging {len(similar_components)} components of {len(centers_dict)} clusters')
    # max_length_component = torch.tensor([len(component) for component in similar_components]).sort(descending=True)[0]
    # Updating centers_dict based on merged_pairs
    skipped_components = []
    for component in tqdm(similar_components, desc='Merging components', total=len(similar_components), ncols=100):

        if len(component) == 1:
            continue
        if len(component) == 2:
            centers_dict, count_dict, label_mapping = merge_pair(component, centers_dict, count_dict, label_mapping)
        elif len(component) > 10000:
            skipped_components.append(component)
            continue
        else:
            centers_dict, count_dict, label_mapping = merge_component(component, centers_dict, count_dict, label_mapping, inner_threshold)
        
    return centers_dict, count_dict, label_mapping, skipped_components



def merge_component(component, centers_dict, count_dict, label_mapping, inner_threshold=0.7):
    n_centers = len(centers_dict)
    do_merge = True
    sub_center_dict = {label: centers_dict[label] for label in component}
    sub_count_dict = {label: count_dict[label] for label in component}
    merge_pairs = []
    while do_merge:
        sub_center_dict, sub_count_dict, merge_pair_idx =  find_and_merge_most_similar_indice_once(sub_center_dict, sub_count_dict, inner_threshold)
        do_merge = merge_pair_idx is not None
        if do_merge:
            merge_pairs.append(merge_pair_idx)
    merge_pairs = torch.tensor(merge_pairs)

    G = nx.Graph()
    G.add_edges_from(merge_pairs.tolist())
    component_groups = list(nx.connected_components(G))
    used_key = set()
    for group in component_groups:
        group = list(group)
        dst_key = group[0]
        sub_feature = torch.stack([centers_dict[label] for label in group]).sum(dim=0)
        sub_count = sum([count_dict[label] for label in group])
        del_keys = group[1:]
        centers_dict[dst_key] = sub_feature
        count_dict[dst_key] = sub_count
        for del_key in del_keys:
            del centers_dict[del_key]
            del count_dict[del_key]
            label_mapping[del_key] = dst_key
        used_key.add(dst_key)
        used_key.update(del_keys)
    # remove unused keys
    unused_keys = component - used_key
    for unused_key in unused_keys:
        del centers_dict[unused_key]
        del count_dict[unused_key]
        label_mapping[unused_key] = -1

    # print(f'Chaning {n_centers} to {len(centers_dict)} Merged {len(component)} components into {len(component_groups)} components and deleted {len(unused_keys)} components')
    return centers_dict, count_dict, label_mapping


def find_and_merge_most_similar_indice_once(sub_center_dict, sub_count_dict, threshold=0.7):
    if len(sub_center_dict) <= 1:
        return sub_center_dict, sub_count_dict, None

    labels = list(sub_center_dict.keys())
    component_features = torch.stack([sub_center_dict[label] for label in labels])
    if len(component_features) > 10000:
        device = 'cpu'
    else:
        device = 'cuda'
    component_features = component_features.to(device)
    component_features = component_features / component_features.norm(dim=1, keepdim=True)
    similarity_matrix =  component_features @ component_features.T
    # print(f'similarity_matrix: {similarity_matrix.shape} {similarity_matrix.min()} {similarity_matrix.max()}')
    # find the most similar pair indice
    most_similar_pair_indice = torch.argmax(similarity_matrix - torch.eye(len(component_features), device=device))
    most_similar_pair_indice = torch.unravel_index(most_similar_pair_indice, similarity_matrix.shape)
    similarity_score = similarity_matrix[most_similar_pair_indice]
    if similarity_score < threshold:
        return sub_center_dict, sub_count_dict, None
    dummy_label_mappging = {label: label for label in sub_center_dict}
    pair_indice = (labels[most_similar_pair_indice[0]], labels[most_similar_pair_indice[1]])
    sub_center_dict, sub_count_dict, dummy_label_mappging = merge_pair(pair_indice, sub_center_dict, sub_count_dict, dummy_label_mappging)
    return sub_center_dict, sub_count_dict, pair_indice


def merge_pair(pair, centers_dict, count_dict, label_mapping):
    assert len(pair) == 2
    key1, key2 = pair

    if key1 not in centers_dict or key2 not in centers_dict:
        return centers_dict, count_dict, label_mapping

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


if __name__ == '__main__':

    root = '/hdd3/data/faces/webface260m/WebFaceV2_subset_by_face_score_v1/regroup_result_v2/vit_base_kprpe_webface12m/raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320/cluster_eps:0.7_min_samples:2_max_group_per_label:-1/facefilter_0.95_merge_threshold_0.7_ambiguous_threshold_0.6_val_sim_threshold_0.7_debug'
    centers_dict = torch.load(os.path.join(root, 'centers.pth'))
    count_dict = torch.load(os.path.join(root, 'count_dict.pth'))
    label_mapping = torch.load(os.path.join(root, 'label_mapping.pth'))

    similar_pairs = torch.load(os.path.join(root, 'similar_pairs.pth'))
    ambiguous_pairs = torch.load(os.path.join(root, 'ambiguous_pairs.pth'))
    to_remove_labels = torch.load(os.path.join(root, 'to_remove_labels.pth'))
    similar_components = torch.load(os.path.join(root, 'similar_components.pth'))

    orig_centers_dict = centers_dict.copy()
    orig_count_dict = count_dict.copy()
    print(f'Before merging: {len(centers_dict)}')
    centers_dict, count_dict, label_mapping = merge_with_precluster_(similar_components, centers_dict, count_dict, label_mapping, inner_threshold=0.7)
    print(f'After merging: {len(centers_dict)}')
    intersecting_keys = set(centers_dict.keys()) & set(orig_centers_dict.keys())
    sub_keys = list(set(intersecting_keys))[:100]
    orig_features = torch.stack([orig_centers_dict[key] for key in sub_keys])
    new_features = torch.stack([centers_dict[key] for key in sub_keys])
    orig_features = orig_features / orig_features.norm(dim=1, keepdim=True)
    new_features = new_features / new_features.norm(dim=1, keepdim=True)
    orig_sim_mat = orig_features @ orig_features.T
    new_sim_mat = new_features @ new_features.T
    

    from cleaning_helper import find_similar_pairs
    similar_pairs, ambiguous_pairs = find_similar_pairs(centers_dict, count_dict, batch_size=9048, similarity_threshold=0.7, ambiguous_threshold=0.6)
    similar_pairs2, ambiguous_pairs2 = find_similar_pairs(orig_centers_dict, orig_count_dict, batch_size=9048, similarity_threshold=0.7, ambiguous_threshold=0.6)

    print(len(similar_pairs))
    print(len(similar_pairs2))
    print(len(ambiguous_pairs))
    print(len(ambiguous_pairs2))