from general_utils.img_utils import stack_images
import cv2
import os
import numpy as np


def plot_pairs(merged_pairs, img_reader, cluster_df, save_dir='/mckim/temp/merged_pairs'):

    for merged_pair in merged_pairs:
        l1 ,l2 = merged_pair
        cluster_df_l1 = cluster_df[cluster_df['label'] == l1]
        cluster_df_l2 = cluster_df[cluster_df['label'] == l2]
        img_paths_l1 = cluster_df_l1['global_path'].values
        img_paths_l2 = cluster_df_l2['global_path'].values

        img_l1 = [img_reader.read_by_path(img_path)[0] for img_path in img_paths_l1]
        img_l2 = [img_reader.read_by_path(img_path)[0] for img_path in img_paths_l2]
        img_l1_resized = [cv2.resize(img, dsize=(112, 112)) for img in img_l1]
        img_l2_resized = [cv2.resize(img, dsize=(112, 112)) for img in img_l2]
        vis1 = stack_images(img_l1_resized, num_cols=5, num_rows=5).astype(np.uint8)
        vis2 = stack_images(img_l2_resized, num_cols=5, num_rows=5).astype(np.uint8)
        vis = cv2.hconcat([vis1, vis2])
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, f'{l1}_{l2}.png'), vis[:, :, ::-1])

