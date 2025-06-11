import pandas as pd
import numpy as np
import os

def print_dataset_stats(df, name="Dataset"):
    print(f"--- {name} Statistics ---")
    print(f"Number of subjects: {df['label'].nunique()}")
    print(f"Number of images: {len(df)}")
    print(f"Average images per subject: {len(df) / df['label'].nunique():.2f}")
    print("---")

if __name__ == '__main__':

    # set seed
    np.random.seed(42)
    root = '/mckim/temp'
    save_root = '/hdd4/data/faces/webface260m/WebFaceV2_subset_by_face_score_v1/whole_body_bundle/raw_img_parquets_640_yolo_cropped/cluster_eps:0.7_min_samples:2_max_group_per_label:1/facefilter_0.95_merge_threshold_0.7_ambiguous_threshold_0.6_val_sim_threshold_0.7_image_size_384'
    save_root = os.path.join(save_root, 'subset_config')
    os.makedirs(save_root, exist_ok=True)
    
    # pd.read_csv(os.path.join(save_root, 'testing.tsv'), sep='\t', header=None)

    tsv_path = os.path.join(root, 'train.tsv')
    shape = os.path.join(root, 'shapes.csv')
    keypoints = os.path.join(root, 'kps_xyn.csv')

    shape_df = pd.read_csv(shape, index_col=0)
    keypoints_df = pd.read_csv(keypoints, index_col=0)
    keypoint_cols = keypoints_df.columns[2:]
    kps_df = (keypoints_df[keypoint_cols])
    kps_df = kps_df.astype(float)
    kps_df_visiblility = kps_df > 0

    df = pd.read_csv(tsv_path, sep='\t', header=None)
    df.columns = ['idx','path', 'label']
    print_dataset_stats(df, "Original Dataset")

    # find subjects with partial face and whole body
    # 0: Nose 1: Left Eye 2: Right Eye 3: Left Ear 4: Right Ear 5: Left Shoulder 6: Right Shoulder 7: Left Elbow
    # 8: Right Elbow 9: Left Wrist 10: Right Wrist 11: Left Hip 12: Right Hip 13:
    # Left Knee 14: Right Knee 15: Left Ankle 16: Right Ankle
    whole_body_colnames = keypoint_cols[20:]
    face_colnames = keypoint_cols[:10]
    whole_body_visible = kps_df_visiblility[whole_body_colnames].any(axis=1)
    face_partial_visible = ~kps_df_visiblility[face_colnames].all(axis=1)
    whole_body_partial_face_kps_df = kps_df[whole_body_visible & face_partial_visible]
    whole_body_partial_face_df = df.loc[whole_body_partial_face_kps_df.index]
    whole_body_partial_face_labels = whole_body_partial_face_df['label'].value_counts()
    # whole_body_partial_face_labels > 30 & whole_body_partial_face_labels < 40
    whole_body_partial_face_labels_usable = whole_body_partial_face_labels[(whole_body_partial_face_labels > 30) & (whole_body_partial_face_labels < 70)]
    whole_body_partial_face_labels_usable_labels = whole_body_partial_face_labels_usable.index
    actual_label_count = df[df['label'].isin(whole_body_partial_face_labels_usable_labels)]['label'].value_counts()
    # find actual label count > 30 & < 70
    actual_label_count_usable = actual_label_count[(actual_label_count > 30) & (actual_label_count < 150)]
    print(f"Number of subjects with partial face and whole body: {len(actual_label_count)}")
    print(f"Number of subjects with partial face and whole body: {len(actual_label_count_usable)}")
    # subset 5000 from ussable subjects
    testing_labels = np.random.choice(actual_label_count_usable.index, size=4000, replace=False)
    testing_df = df[df['label'].isin(testing_labels)]
    print_dataset_stats(testing_df, "Testing Set")
    testing_df.to_csv(os.path.join(save_root, 'testing.tsv'), sep='\t', header=False, index=False)

    # take out 5000 subjects for testing set
    # labels = df['label'].unique()
    # testing_labels = np.random.choice(labels, size=5000, replace=False)
    # testing_df = df[df['label'].isin(testing_labels)]
    # print_dataset_stats(testing_df, "Testing Set")
    # kps_df_visiblility.loc[testing_df.index].mean(axis=0)

    df = df[~df['label'].isin(testing_labels)]
    print_dataset_stats(df, "Training Set")
    df.to_csv(os.path.join(save_root, 'training.tsv'), sep='\t', header=False, index=False)
    # subset to top 25% most frequent labels
    subset_df = df[df['label'].isin(df['label'].value_counts().nlargest(int(df['label'].nunique() * 0.48)).index)]
    print_dataset_stats(subset_df, "Subset of Top 50% Labels")

    # random 70% subset of Frequent subset (subset the label)
    labels = subset_df['label'].unique()
    keep_labels = np.random.choice(labels, size=int(len(labels) * 0.70), replace=False)
    subset_df_v2 = subset_df[subset_df['label'].isin(keep_labels)]
    print_dataset_stats(subset_df_v2, "Random 70% Subset of Frequent Subset")
    subset_df_v2.to_csv(os.path.join(save_root, 'subset_12m.tsv'), sep='\t', header=False, index=False)
    # random 25% subset of Frequent subset (subset the label)
    labels = subset_df['label'].unique()
    keep_labels = np.random.choice(labels, size=int(len(labels) * 0.25), replace=False)
    subset_df_v2 = subset_df[subset_df['label'].isin(keep_labels)]
    print_dataset_stats(subset_df_v2, "Random 50% Subset of Frequent Subset")
    subset_df_v2.to_csv(os.path.join(save_root, 'subset_4m.tsv'), sep='\t', header=False, index=False)

    # random 25% of the whole dataset
    labels = df['label'].unique()
    keep_labels = np.random.choice(labels, size=int(len(labels) * 0.23), replace=False)
    subset_df_v3 = df[df['label'].isin(keep_labels)]
    print_dataset_stats(subset_df_v3, "Random 25% of Whole Dataset")
    subset_df_v3.to_csv(os.path.join(save_root, 'subset_4m_random.tsv'), sep='\t', header=False, index=False)
