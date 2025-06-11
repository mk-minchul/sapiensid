import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)
import os, sys
sys.path.insert(0, str(root))
import numpy as np
np.bool = np.bool_  # fix bug for mxnet 1.9.1
np.object = np.object_
np.float = np.float_

import datetime
from reader import SplittedFeatureRecordReader
from reader_img import SplittedRecordReader as SplittedRecordReaderImg
import pathlib
import argparse
from general_utils.os_utils import get_all_files
import torch
import pandas as pd
import lovely_tensors as lt
lt.monkey_patch()
from tqdm import tqdm
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from center_helper import compute_center
from cleaning_helper import merge_, delete_, find_duplicate_images, find_similar_pairs
from merge_helper import merge_with_precluster_
from vis_helper import plot_pairs
from val_dataset_helper import find_too_sim_to_val_center
import cv2
import torch
import networkx as nx
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_record_root', type=str, default='/ssd2/data/faces/sapiensid_webbody/raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320')
    parser.add_argument('--feature_record_root', type=str, default='/ssd2/data/faces/sapiensid_webbody/face_features/vit_base_kprpe_webface12m/raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320')
    parser.add_argument('--cluster_result_path', type=str, default='/ssd2/data/faces/sapiensid_webbody/cluster_result/vit_base_kprpe_webface12m/raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320/cluster_eps:0.7_min_samples:2_max_group_per_label:1.csv')
    parser.add_argument('--face_score_filtering_threshold', type=float, default=0.95)
    parser.add_argument('--merge_threshold', type=float, default=0.7)
    parser.add_argument('--ambiguous_threshold', type=float, default=0.6)
    parser.add_argument('--val_sim_threshold', type=float, default=0.7)
    parser.add_argument('--duplicate_sim_threshold', type=float, default=0.9)
    parser.add_argument('--use_face_score_for_merge', action='store_true')
    parser.add_argument('--merge_method', type=str, default='precluster', choices=['precluster', 'simple'])
    parser.add_argument('--do_plot', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--save_name', type=str, default='debug')

    args = parser.parse_args()

    assert 'face_features' in args.feature_record_root
    today_date = datetime.datetime.now().strftime('%Y%m%d-%H%M')
    save_root = args.cluster_result_path.replace('cluster_result', 'regroup_result_v2')

    # save_root = os.path.join(save_root.replace('.csv', ''), args.save_name + '_' + today_date)
    save_root = os.path.join(save_root.replace('.csv', ''), args.save_name)
    os.makedirs(save_root, exist_ok=True)
    with open(os.path.join(save_root, 'args.yaml'), 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')
    print('save_root', save_root)

    # load feature data
    assert os.path.exists(args.feature_record_root), f'{args.feature_record_root} not found'
    feature_record_paths = get_all_files(args.feature_record_root, extension_list=['.rec'], sort=True)
    print('Found records:', len(feature_record_paths))
    feature_record_paths = [os.path.dirname(path) for path in feature_record_paths]
    if args.debug:
        feature_record_paths = feature_record_paths[:1]
        parquet_scopes = [p.split('/')[-1].split('_')[-1].split('-') for p in feature_record_paths]
        parquet_scopes = [np.arange(int(bin[0]), int(bin[1])+1) for bin in parquet_scopes]
        parquet_scopes = ["/"+ str(item) + '_parquet' for sublist in parquet_scopes for item in sublist]

    feature_dataset = SplittedFeatureRecordReader(feature_record_paths)
    sample, label, path = feature_dataset.read_by_index(1)

    with open(os.path.join(save_root, 'stats.csv'), 'w') as f:
        # header: stage, n_subjects, n_images
        f.write('stage,n_subjects,n_images\n')

    # load cluster result
    cluster_df = pd.read_csv(args.cluster_result_path)
    print('initial cluster df', cluster_df.shape)
    if args.debug:
        feature_available_paths = set(cluster_df['global_path']).intersection(set(feature_dataset.paths))
        cluster_df = cluster_df[cluster_df['global_path'].isin(feature_available_paths)]
        print('reducing cluster df to', len(cluster_df), 'images')
    
    cluster_df['remapped_label'] = cluster_df['label']
    cluster_df.to_csv(os.path.join(save_root, 'cluster_df_initial.csv'), index=False)
    with open(os.path.join(save_root, 'stats.csv'), 'a') as f:
        n_subjects = len(cluster_df['remapped_label'].unique())
        n_images = len(cluster_df)
        f.write(f'initial cluster df, {n_subjects}, {n_images}\n')

    # optionally read images to visualize
    if args.do_plot:
        img_record_paths = get_all_files(args.img_record_root, extension_list=['.rec'], sort=True)
        img_record_paths = [os.path.dirname(path) for path in img_record_paths]
        if args.debug:
            img_record_paths = [path for path in img_record_paths if any(parquet in path for parquet in parquet_scopes)]
        img_reader = SplittedRecordReaderImg(img_record_paths, return_img=True)
        transform = Compose([ToTensor(),
                             Resize((112, 112), antialias=True),
                             Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])
    else:
        img_reader = None

    # load face scores
    face_scores_paths = get_all_files(args.img_record_root, extension_list=['.csv'], sort=True)
    face_scores_paths = [path for path in face_scores_paths if 'face_confidence' in path]
    if args.debug:
        face_scores_paths = [path for path in face_scores_paths if any(parquet in path for parquet in parquet_scopes)]
    face_scores_df = []
    for face_score_path in tqdm(face_scores_paths, total=len(face_scores_paths), desc='Loading face scores'):
        face_score = pd.read_csv(face_score_path) # ['local_idx', 'global_idx', 'detection_idx', 'confidence']
        tsv_path = face_score_path.replace('face_confidence.csv', 'train.tsv')
        tsv = pd.read_csv(tsv_path, sep='\t', header=None)
        tsv.columns = ['local_idx', 'path', 'label']
        face_score = pd.merge(face_score, tsv[['local_idx', 'path']], on='local_idx', how='left')
        face_score = face_score[['path', 'confidence']]
        face_scores_df.append(face_score)
    face_scores_df = pd.concat(face_scores_df)
    face_scores_df.set_index('path', inplace=True)
    cluster_df['face_score'] = cluster_df['global_path'].apply(lambda x: face_scores_df.loc[x, 'confidence'])
    cluster_df = cluster_df[cluster_df['face_score'] > args.face_score_filtering_threshold]
    cluster_df.loc[:, 'count'] = cluster_df['remapped_label'].groupby(cluster_df['remapped_label']).transform('count')
    cluster_df = cluster_df[cluster_df['count'] >= 2]
    cluster_df.to_csv(os.path.join(save_root, 'cluster_df_with_face_score_filter.csv'), index=False)
    with open(os.path.join(save_root, 'stats.csv'), 'a') as f:
        n_subjects = len(cluster_df['remapped_label'].unique())
        n_images = len(cluster_df)
        f.write(f'after face score filtering, {n_subjects}, {n_images}\n')

    # compute center
    if os.path.exists(os.path.join(save_root, 'centers.pth')):
        centers_dict = torch.load(os.path.join(save_root, 'centers.pth'))
        count_dict = torch.load(os.path.join(save_root, 'count_dict.pth'))
        label_mapping = torch.load(os.path.join(save_root, 'label_mapping.pth'))
    else:
        centers_dict, count_dict, label_mapping = compute_center(feature_dataset, cluster_df,
                                                                 args.use_face_score_for_merge,
                                                                 img_reader,
                                                                 use_cuda=args.use_cuda, do_plot=args.do_plot)
        torch.save(centers_dict, os.path.join(save_root, 'centers.pth'))
        torch.save(count_dict, os.path.join(save_root, 'count_dict.pth'))
        torch.save(label_mapping, os.path.join(save_root, 'label_mapping.pth'))
    
    def remap(x):
        return x if x not in label_mapping else label_mapping[x]
    cluster_df['remapped_label'] = cluster_df['remapped_label'].apply(remap)
    cluster_df = cluster_df[cluster_df['remapped_label'] != -1]
    cluster_df.loc[:, 'count'] = cluster_df['remapped_label'].groupby(cluster_df['remapped_label']).transform('count')
    cluster_df = cluster_df[cluster_df['count'] >= 2]
    cluster_df.to_csv(os.path.join(save_root, 'cluster_df_after_compute_center.csv'), index=False)
    with open(os.path.join(save_root, 'stats.csv'), 'a') as f:
        n_subjects = len(cluster_df['remapped_label'].unique())
        n_images = len(cluster_df)
        f.write(f'after compute center, {n_subjects}, {n_images}\n')
    using_labels = cluster_df['remapped_label'].values
    centers_dict = {label: centers_dict[label] for label in using_labels if label != -1}
    count_dict = {label: count_dict[label] for label in using_labels if label != -1}

    # find similar pairs
    if not os.path.exists(os.path.join(save_root, 'similar_pairs.pth')):
        similar_pairs, ambiguous_pairs = find_similar_pairs(centers_dict, count_dict, args.batch_size, args.merge_threshold, args.ambiguous_threshold)
        to_remove_labels = [pair[0] for pair in ambiguous_pairs]
        torch.save(similar_pairs, os.path.join(save_root, 'similar_pairs.pth'))
        torch.save(ambiguous_pairs, os.path.join(save_root, 'ambiguous_pairs.pth'))
        torch.save(to_remove_labels, os.path.join(save_root, 'to_remove_labels.pth'))
    else:
        similar_pairs = torch.load(os.path.join(save_root, 'similar_pairs.pth'))
        ambiguous_pairs = torch.load(os.path.join(save_root, 'ambiguous_pairs.pth'))
        to_remove_labels = torch.load(os.path.join(save_root, 'to_remove_labels.pth'))


    G = nx.Graph()
    G.add_edges_from(similar_pairs)
    components = list(nx.connected_components(G))
    torch.save(components, os.path.join(save_root, 'similar_components.pth'))
    print(f'found {len(components)} components')
    if len(components) != 0:
        print(f'largest component has {max([len(c) for c in components])} images')
        print(f'smallest component has {min([len(c) for c in components])} images')

    # merge
    if os.path.exists(os.path.join(save_root, 'label_mapping_after_merge.pth')):
        print('loading from existing merge...')
        label_mapping = torch.load(os.path.join(save_root, 'label_mapping_after_merge.pth'))
        centers_dict = torch.load(os.path.join(save_root, 'centers_after_merge.pth'))
        count_dict = torch.load(os.path.join(save_root, 'count_dict_after_merge.pth'))
        skipped_components = torch.load(os.path.join(save_root, 'skipped_components.pth'))
    else:
        print('merging...')
        if args.merge_method == 'precluster':
            centers_dict, count_dict, label_mapping, skipped_components = merge_with_precluster_(components, centers_dict, count_dict, label_mapping, inner_threshold=args.merge_threshold)
        else:
            centers_dict, count_dict, label_mapping = merge_(similar_pairs, centers_dict, count_dict, label_mapping)
            skipped_components = []
        torch.save(label_mapping, os.path.join(save_root, 'label_mapping_after_merge.pth'))
        torch.save(centers_dict, os.path.join(save_root, 'centers_after_merge.pth'))
        torch.save(count_dict, os.path.join(save_root, 'count_dict_after_merge.pth'))
        torch.save(skipped_components, os.path.join(save_root, 'skipped_components.pth'))
        print('after merge, len(centers_dict)', len(centers_dict))

    def remap(x):
        return x if x not in label_mapping else label_mapping[x]
    cluster_df['remapped_label'] = cluster_df['remapped_label'].apply(remap)
    cluster_df = cluster_df[cluster_df['remapped_label'] != -1]
    cluster_df.loc[:, 'count'] = cluster_df['remapped_label'].groupby(cluster_df['remapped_label']).transform('count')
    cluster_df = cluster_df[cluster_df['count'] >= 2]
    cluster_df.to_csv(os.path.join(save_root, 'cluster_df_after_merge.csv'), index=False)
    with open(os.path.join(save_root, 'stats.csv'), 'a') as f:
        n_subjects = len(cluster_df['remapped_label'].unique())
        n_images = len(cluster_df)
        f.write(f'after merge, {n_subjects}, {n_images}\n')

    # merge skipped components
    print('merging skipped components...')
    skipped_component_flat = set([item for sublist in skipped_components for item in sublist])
    skipped_merge_pairs = []
    for pair in similar_pairs:
        if pair[0] in skipped_component_flat or pair[1] in skipped_component_flat:
            skipped_merge_pairs.append(pair)
    torch.save(skipped_merge_pairs, os.path.join(save_root, 'skipped_merge_pairs.pth'))
    print('skipped_merge_pairs', len(skipped_merge_pairs))

    centers_dict, count_dict, label_mapping = merge_(skipped_merge_pairs, centers_dict, count_dict, label_mapping)
    torch.save(label_mapping, os.path.join(save_root, 'label_mapping_after_merge_skipped.pth'))
    torch.save(centers_dict, os.path.join(save_root, 'centers_after_merge_skipped.pth'))
    torch.save(count_dict, os.path.join(save_root, 'count_dict_after_merge_skipped.pth'))

    def remap(x):
        return x if x not in label_mapping else label_mapping[x]
    cluster_df['remapped_label'] = cluster_df['remapped_label'].apply(remap)
    cluster_df = cluster_df[cluster_df['remapped_label'] != -1]
    cluster_df.loc[:, 'count'] = cluster_df['remapped_label'].groupby(cluster_df['remapped_label']).transform('count')
    cluster_df = cluster_df[cluster_df['count'] >= 2]
    cluster_df.to_csv(os.path.join(save_root, 'cluster_df_after_merge_skipped.csv'), index=False)
    with open(os.path.join(save_root, 'stats.csv'), 'a') as f:
        n_subjects = len(cluster_df['remapped_label'].unique())
        n_images = len(cluster_df)
        f.write(f'after merge skipped, {n_subjects}, {n_images}\n')

    # delete
    print('deleting...')
    centers_dict, count_dict, label_mapping = delete_(to_remove_labels, centers_dict, count_dict, label_mapping)
    torch.save(label_mapping, os.path.join(save_root, 'label_mapping_after_delete.pth'))
    torch.save(centers_dict, os.path.join(save_root, 'centers_after_delete.pth'))
    torch.save(count_dict, os.path.join(save_root, 'count_dict_after_delete.pth'))

    def remap(x):
        return x if x not in label_mapping else label_mapping[x]
    cluster_df['remapped_label'] = cluster_df['remapped_label'].apply(remap)
    cluster_df = cluster_df[cluster_df['remapped_label'] != -1]
    cluster_df.loc[:, 'count'] = cluster_df['remapped_label'].groupby(cluster_df['remapped_label']).transform('count')
    cluster_df = cluster_df[cluster_df['count'] >= 2]
    cluster_df.to_csv(os.path.join(save_root, 'cluster_df_after_delete.csv'), index=False)
    print('after delete, len(centers_dict)', len(centers_dict))
    with open(os.path.join(save_root, 'stats.csv'), 'a') as f:
        n_subjects = len(cluster_df['remapped_label'].unique())
        n_images = len(cluster_df)
        f.write(f'after delete, {n_subjects}, {n_images}\n')

    if args.do_plot:
        plot_pairs(similar_pairs, img_reader, cluster_df, save_dir='./to_merge_pairs_v3')
        plot_pairs(ambiguous_pairs, img_reader, cluster_df, save_dir='./ambiguous_pairs_v3')

    # valset delete
    print('valset deleting...')
    if pathlib.Path(os.path.join('./', 'combined_val_centers.pth')).exists():
        val_center = torch.load(os.path.join('./', 'combined_val_centers.pth'))
    else:
        raise ValueError('combined_val_centers.pth not found')

    # find too sim to val center
    too_sim_to_val_center = find_too_sim_to_val_center(centers_dict, val_center, args.val_sim_threshold, args.use_cuda)
    centers_dict, count_dict, label_mapping = delete_(too_sim_to_val_center, centers_dict, count_dict, label_mapping)
    torch.save(label_mapping, os.path.join(save_root, 'label_mapping_after_val_delete.pth'))
    torch.save(centers_dict, os.path.join(save_root, 'centers_after_val_delete.pth'))
    torch.save(count_dict, os.path.join(save_root, 'count_dict_after_val_delete.pth'))
    print('after val delete, len(centers_dict)', len(centers_dict))


    def remap(x):
        return x if x not in label_mapping else label_mapping[x]
    cluster_df['remapped_label'] = cluster_df['remapped_label'].apply(remap)
    cluster_df = cluster_df[cluster_df['remapped_label'] != -1]
    cluster_df.loc[:, 'count'] = cluster_df['remapped_label'].groupby(cluster_df['remapped_label']).transform('count')
    cluster_df = cluster_df[cluster_df['count'] >= 2]
    cluster_df.to_csv(os.path.join(save_root, 'cluster_df_after_val_delete.csv'), index=False)
    with open(os.path.join(save_root, 'stats.csv'), 'a') as f:
        n_subjects = len(cluster_df['remapped_label'].unique())
        n_images = len(cluster_df)
        f.write(f'after val delete, {n_subjects}, {n_images}\n')

    print('finding duplicate images...')
    duplicate_pairs = find_duplicate_images(cluster_df, feature_dataset, similarity_threshold=args.duplicate_sim_threshold, use_cuda=args.use_cuda)
    torch.save(duplicate_pairs, os.path.join(save_root, 'duplicate_pairs.pth'))
    print('duplicate_images', len(duplicate_pairs))
    if args.do_plot:
        for pair in duplicate_pairs:
            img1 = img_reader.read_by_path(pair[0])[0]
            img2 = img_reader.read_by_path(pair[1])[0]
            img1 = cv2.resize(img1, (112, 112))
            img2 = cv2.resize(img2, (112, 112))
            vis = np.concatenate([img1, img2], axis=1)[:, :, ::-1]
            os.makedirs(os.path.join('/mckim/temp', 'duplicate_pairs_v3'), exist_ok=True)
            cv2.imwrite(os.path.join('/mckim/temp', 'duplicate_pairs_v3', f'{pair[0].split("/")[-1]}_{pair[1].split("/")[-1]}'), vis)

    # remove duplicate images
    to_delete = set([pair[1] for pair in duplicate_pairs])
    cluster_df = cluster_df[~cluster_df['global_path'].isin(to_delete)]
    cluster_df.to_csv(os.path.join(save_root, 'cluster_df_after_duplicate_delete.csv'), index=False)
    with open(os.path.join(save_root, 'stats.csv'), 'a') as f:
        n_subjects = len(cluster_df['remapped_label'].unique())
        n_images = len(cluster_df)
        f.write(f'after duplicate delete, {n_subjects}, {n_images}\n')

    # remove cluster with less than 2 images
    print('removing cluster with less than 2 images...')
    cluster_df.loc[:, 'count'] = cluster_df['remapped_label'].groupby(cluster_df['remapped_label']).transform('count')
    cluster_df = cluster_df[cluster_df['count'] >= 2]
    cluster_df.to_csv(os.path.join(save_root, 'cluster_final.csv'), index=False)
    with open(os.path.join(save_root, 'stats.csv'), 'a') as f:
        n_subjects = len(cluster_df['remapped_label'].unique())
        n_images = len(cluster_df)
        f.write(f'final after remove cluster with less than 2 images, {n_subjects}, {n_images}\n')

    # remove images with low face score
    print('removing images with low face score...')
    cluster_df['_global_path'] = cluster_df['global_path']
    cluster_df.set_index('_global_path', inplace=True)
    cluster_df_only_high_face_score = cluster_df[face_scores_df['confidence'] > args.face_score_filtering_threshold]
    cluster_df_only_high_face_score.reset_index(inplace=True)
    # remove images with less than 2 images
    cluster_df_only_high_face_score.loc[:, 'count'] = cluster_df_only_high_face_score['remapped_label'].groupby(cluster_df_only_high_face_score['remapped_label']).transform('count')
    cluster_df_only_high_face_score = cluster_df_only_high_face_score[cluster_df_only_high_face_score['count'] >= 2]
    with open(os.path.join(save_root, 'stats.csv'), 'a') as f:
        n_subjects = len(cluster_df_only_high_face_score['remapped_label'].unique())
        n_images = len(cluster_df_only_high_face_score)
        f.write(f'final after remove images with low face score, {n_subjects}, {n_images}\n')

    # save cluster df
    print('saving cluster df...')
    cluster_df_only_high_face_score.reset_index(inplace=True)
    cluster_df_only_high_face_score.drop(columns=['_global_path', 'index'], inplace=True)
    cluster_df_only_high_face_score.to_csv(os.path.join(save_root, 'cluster_final_only_high_face_score.csv'), index=False)
    with open(os.path.join(save_root, 'stats.csv'), 'a') as f:
        n_subjects = len(cluster_df_only_high_face_score['remapped_label'].unique())
        n_images = len(cluster_df_only_high_face_score)
        f.write(f'finalafter remove images with low face score, {n_subjects}, {n_images}\n')

