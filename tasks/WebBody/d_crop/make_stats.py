import os
from general_utils.os_utils import get_all_files
import pandas as pd
from tqdm import tqdm
import numpy as np


if __name__ == '__main__':


    root = '/hdd4/data/faces/webface260m/WebFaceV2_subset_by_face_score_v1/raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320'
    all_tsv = get_all_files(root, extension_list=['.tsv'], sort=True)

    roots = [os.path.dirname(path) for path in all_tsv]

    unique_labels = set()
    unique_image_names = set()
    total_samples = 0

    conf_list = []
    for root_dir in tqdm(roots, total=len(roots)):
        tsv_path = os.path.join(root_dir, 'train.tsv')
        confidence_path = os.path.join(root_dir, 'face_confidence.csv')
        # tsv_df = pd.read_csv(tsv_path, sep='\t', header=None)
        # tsv_df.columns = ['local_index', 'path', 'label']
        confidence_df = pd.read_csv(confidence_path)
        conf_list.extend(confidence_df['confidence'].to_list())

    conf_list = np.array(conf_list)
    print('mean', np.mean(conf_list))
    print('std', np.std(conf_list))
    print('min', np.min(conf_list))
    print('max', np.max(conf_list))
    print('median', np.median(conf_list))
    mean_conf = np.mean(conf_list)

    # plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(conf_list, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(mean_conf, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_conf:.2f}')
    plt.title('Distribution of Face Confidence Scores', fontsize=16, fontweight='bold')
    plt.xlabel('Confidence Score', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('/mckim/temp/face_confidence_distribution.png')
