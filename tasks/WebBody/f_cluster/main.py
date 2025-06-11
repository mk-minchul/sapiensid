import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)
import os, sys
sys.path.append(os.path.join(root))
import numpy as np
np.bool = np.bool_  # fix bug for mxnet 1.9.1
np.object = np.object_
np.float = np.float_

from reader import SplittedFeatureRecordReader
from reader_img import SplittedRecordReader as SplittedRecordReaderImg
import argparse
from general_utils.os_utils import get_all_files
from general_utils.img_utils import tensor_to_pil, put_text_pil, stack_images

from sklearn.cluster import DBSCAN as DBSCAN_CPU
import torch
import pandas as pd
import lovely_tensors as lt
lt.monkey_patch()
import math
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from tqdm import tqdm

# pip install \
#     --extra-index-url=https://pypi.nvidia.com \
#     "cuml-cu12==24.8.*"


def most_frequent_non_negative_label(cluster_labels):
    # Filter out the -1 labels
    filtered_labels = cluster_labels[cluster_labels != -1]

    # Check if there are any valid labels left
    if filtered_labels.size == 0:
        return None

    # Find the most frequent label
    unique_labels, counts = np.unique(filtered_labels, return_counts=True)
    most_frequent_label = unique_labels[np.argmax(counts)]

    return most_frequent_label


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--record_root', type=str, default='/ssd2/data/faces/webface260m/WebFaceV2_subset_by_face_score_v1/face_features/vit_base_kprpe_webface12m/raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320')

    parser.add_argument('--eps', type=float, default=0.7)
    parser.add_argument('--min_samples', type=int, default=2)
    parser.add_argument('--max_group_per_label', type=int, default=-1)

    args = parser.parse_args()

    assert 'face_features' in args.record_root
    save_dir = args.record_root.replace('face_features', 'cluster_result')
    os.makedirs(save_dir, exist_ok=True)
    print('save_dir', save_dir)
    save_path = os.path.join(save_dir, f'cluster_eps:{args.eps}_min_samples:{args.min_samples}_max_group_per_label:{args.max_group_per_label}.csv')
    print('save_path', save_path)

    record_paths = get_all_files(args.record_root, extension_list=['.rec'], sort=True)
    print('Found records:', len(record_paths))
    record_paths = [os.path.dirname(path) for path in record_paths]
    dataset = SplittedFeatureRecordReader(record_paths)
    sample, label, path = dataset.read_by_index(1)

    df = pd.DataFrame(pd.Series(dataset.paths), columns=['path'])
    df['label'] = df['path'].apply(lambda x: x.split('/')[0])
    groupby = df.groupby('label')

    n_labels = 0
    n_images = 0
    csv_writer = open(save_path, 'w')
    csv_writer.write('local_idx,global_path,label\n')
    for label, group in tqdm(groupby, total=len(groupby), desc='Clustering'):
        if len(group) <= 1:
            continue
        image_paths = group['path'].tolist()
        feats = torch.stack([dataset.read_by_path(path)[0] for path in image_paths])
        feats = feats.to('cuda')
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        similarity_matrix_cuda = torch.mm(feats, feats.T)

        # higher eps leads to more samples being same subject I changed from 0.5 to 0.7
        # min_samples is the minimum number of samples in a cluster
        # you can use cuml to speed up the clustering
        # import cuml
        # import cupy as cp
        # dbscan_cuda = cuml.DBSCAN(eps=args.eps, min_samples=args.min_samples, metric='precomputed')
        # cluster_labels_cuda = dbscan_cuda.fit_predict(1 - similarity_matrix_cuda)
        # cluster_labels = cp.asnumpy(cluster_labels_cuda)
        dbscan = DBSCAN_CPU(eps=args.eps, min_samples=args.min_samples, metric='precomputed')
        cluster_labels = dbscan.fit_predict(1 - torch.clip(similarity_matrix_cuda, 0, 1).cpu().numpy())

        filtered_labels = cluster_labels[cluster_labels != -1]
        if filtered_labels.size == 0:
            continue

        unique_labels, counts = np.unique(filtered_labels, return_counts=True)
        counts, unique_labels = zip(*sorted(zip(counts, unique_labels), reverse=True))
        for l_idx, (label, count) in enumerate(zip(unique_labels, counts)):
            if count < 2:
                continue

            if args.max_group_per_label > 0 and l_idx >= args.max_group_per_label:
                break

            label_indices = np.where(cluster_labels == label)[0]
            for i in label_indices:
                csv_writer.write(f'{n_images},{image_paths[i]},{n_labels}\n')
                n_images += 1
            n_labels += 1

    csv_writer.close()
