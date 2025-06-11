import os
from general_utils.os_utils import get_all_files
import pandas as pd
from tqdm import tqdm
import numpy as np


if __name__ == '__main__':


    root = '/hdd3/data/faces/webface260m/WebFaceV2_subset_by_face_score_v1/raw_img_parquets_640_yolo_cropped'
    all_tsv = get_all_files(root, extension_list=['.tsv'], sort=True)

    roots = [os.path.dirname(path) for path in all_tsv]
    # train.tsv confidence.csv body_kps_xyn.csv bbox_xyxyn.csv
    all_tsv_paths = [os.path.join('train.tsv') for path in roots]
    all_confidence_paths = [os.path.join('confidence.csv') for path in roots]

    unique_labels = set()
    unique_image_names = set()
    total_samples = 0

    bbox_columns_x = np.array([[f'kp_{k}_x',] for k in range(17)]).flatten()
    kps_visible_percentages = []

    for root_dir in tqdm(roots, total=len(roots)):
        tsv_path = os.path.join(root_dir, 'train.tsv')
        # confidence_path = os.path.join(root_dir, 'confidence.csv')
        df = pd.read_csv(tsv_path, sep='\t', header=None)
        df.columns = ['local_index', 'path', 'label']
        unique_labels.update(df['label'].unique())
        image_names = df['path'].apply(lambda x:x.split('/')[-1].split('_')[0])
        unique_image_names.update(image_names.unique())
        total_samples += len(df)

        body_kps_xyn_path = os.path.join(root_dir, 'body_kps_xyn.csv')
        body_kps_xyn_df = pd.read_csv(body_kps_xyn_path)
        kps_visible_percentage = (body_kps_xyn_df[bbox_columns_x] > 0).sum(0) / len(body_kps_xyn_df)
        kps_visible_percentage.index = ['Nose',
                                        'Left Eye', 'Right Eye',
                                        'Left Ear', 'Right Ear',
                                        'Left Shoulder', 'Right Shoulder',
                                        'Left Elbow', 'Right Elbow',
                                        'Left Wrist', 'Right Wrist',
                                        'Left Hip', 'Right Hip',
                                        'Left Knee', 'Right Knee',
                                        'Left Ankle', 'Right Ankle']
        kps_visible_percentages.append(kps_visible_percentage)

        # body_bbox_xyxyn_path = os.path.join(root_dir, 'bbox_xyxyn.csv')
        # body_bbox_xyxyn_df = pd.read_csv(body_bbox_xyxyn_path)

        # kps_visible_percentage.to_csv('/mckim/temp/kps_visible_percentage.csv')
        # 0: Nose 1: Left Eye 2: Right Eye 3: Left Ear 4: Right Ear 5: Left Shoulder 6: Right Shoulder 7: Left Elbow
        # 8: Right Elbow 9: Left Wrist 10: Right Wrist 11: Left Hip 12: Right Hip 13:
        # Left Knee 14: Right Knee 15: Left Ankle 16: Right Ankle

    print('total_samples', total_samples)
    kps_visible_percentages_df = pd.concat(kps_visible_percentages, axis=1).T
    kps_visible_percentages_df.mean(axis=0).to_csv('/mckim/temp/kps_visible_percentage_v2.csv')
    print('unique_image_names', len(unique_image_names))
    print('unique_labels', len(unique_labels))