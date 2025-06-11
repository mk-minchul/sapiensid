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

from general_utils.os_utils import get_all_files
import pandas as pd


if __name__ == '__main__':

    root = '/hdd3/data/faces/webface260m/WebFaceV2_subset_by_face_score_v1/cluster_result/vit_base_kprpe_webface12m/raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320'
    cluster_result_paths = get_all_files(root, extension_list=['.csv'], sort=True)

    cluster_result_paths = [
     '/hdd3/data/faces/webface260m/WebFaceV2_subset_by_face_score_v1/cluster_result/vit_base_kprpe_webface12m/raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320/cluster_eps:0.7_min_samples:2_max_group_per_label:1.csv',
     '/hdd3/data/faces/webface260m/WebFaceV2_subset_by_face_score_v1/cluster_result/vit_base_kprpe_webface12m/raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320/cluster_eps:0.7_min_samples:2_max_group_per_label:-1.csv',
     '/hdd3/data/faces/webface260m/WebFaceV2_subset_by_face_score_v1/cluster_result/vit_base_kprpe_webface12m/raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320/cluster_eps:0.65_min_samples:2_max_group_per_label:-1.csv',
     '/hdd3/data/faces/webface260m/WebFaceV2_subset_by_face_score_v1/cluster_result/vit_base_kprpe_webface12m/raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320/cluster_eps:0.75_min_samples:2_max_group_per_label:-1.csv']



    stats = []
    for cluster_result_path in cluster_result_paths:

        name = os.path.basename(cluster_result_path)
        df = pd.read_csv(cluster_result_path)
        unique_labels = df['label'].unique()
        stat = {
            'name': name,
            'num_unique_labels': len(unique_labels),
            'num_samples': len(df),
            'num_samples_per_label': df.groupby('label').size().mean(),
        }
        stats.append(stat)
    stats = pd.DataFrame(stats)
    stats.to_csv('/mckim/temp/cluster_stat.csv', index=False)
