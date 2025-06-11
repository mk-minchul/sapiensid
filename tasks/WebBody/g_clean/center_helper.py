import torch
from tqdm import tqdm

def compute_center(feature_dataset, cluster_df, use_face_score_for_merge, img_reader, do_plot=False, use_cuda=True):
    cluster_label_groupby = cluster_df.groupby('remapped_label')

    centers = {}
    count_dict = {}
    label_mapping = {}
    num_subjects_removed_by_face_score = 0
    for label, group in tqdm(cluster_label_groupby, desc='compute_center', total=len(cluster_label_groupby), ncols=75, miniters=10):
        image_paths = group['global_path'].values
        face_scores = group['face_score'].values

        if len(image_paths) <= 1:
            num_subjects_removed_by_face_score += 1
            label_mapping[label] = -1
            continue

        features = torch.stack([feature_dataset.read_by_path(img_path)[0] for img_path in image_paths])
        if use_cuda:
            features = features.cuda()
        if use_face_score_for_merge:
            face_scores = torch.tensor(face_scores, device=features.device, dtype=features.dtype)
            # face_scores = face_scores / face_scores.sum()
        else:
            face_scores = torch.ones(len(image_paths), device=features.device)

        if do_plot:
            images = [img_reader.read_by_path(img_path)[0] for img_path in image_paths]
            # do plot here

        avg_feature = (features * face_scores[:, None]).sum(0).cpu()
        centers[label] = avg_feature
        count_dict[label] = len(image_paths)

    print(f'num_subjects_removed_by_face_score: {num_subjects_removed_by_face_score}')
    return centers, count_dict, label_mapping
