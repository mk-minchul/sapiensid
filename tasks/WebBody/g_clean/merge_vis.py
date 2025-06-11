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
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from general_utils.os_utils import get_all_files
from reader import SplittedFeatureRecordReader
from reader_img import SplittedRecordReader as SplittedRecordReaderImg
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import cv2
from tqdm import tqdm

def main():
    # load data
    root = '/ssd2/data/faces/webface260m/WebFaceV2_subset_by_face_score_v1/regroup_result_v2/vit_base_kprpe_webface12m/raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320/cluster_eps:0.7_min_samples:2_max_group_per_label:-1/facefilter_0.95_merge_threshold_0.7_ambiguous_threshold_0.6_val_sim_threshold_0.7'
    cluster_df = pd.read_csv(os.path.join(root, 'cluster_df_after_compute_center.csv'))
    # centers_after_merge = torch.load(os.path.join(root, 'centers_after_merge.pth'))
    count_dict_after_merge = torch.load(os.path.join(root, 'count_dict_after_merge.pth'))
    label_mapping = torch.load(os.path.join(root, 'label_mapping_after_merge.pth'))
    cluster_df_orig = cluster_df.copy()
    orig_n_subjects = len(cluster_df_orig['remapped_label'].unique())
    orig_n_images = len(cluster_df_orig)


    # load image
    img_record_root = '/hdd4/data/faces/webface260m/WebFaceV2_subset_by_face_score_v1/raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320'
    img_record_paths = get_all_files(img_record_root, extension_list=['.rec'], sort=True)
    img_record_paths = [os.path.dirname(path) for path in img_record_paths]
    img_record_paths = img_record_paths[:10]
    img_reader = SplittedRecordReaderImg(img_record_paths, return_img=True)
    transform = Compose([ToTensor(),
                            Resize((112, 112), antialias=True),
                            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])

    # find merging samples
    mapping_df = pd.DataFrame(pd.Series(label_mapping), columns=['remapped_label'])
    groupby = mapping_df.groupby('remapped_label').apply(lambda x: x.index.tolist())
    groupby_list = groupby.to_dict()
    groupby_list = {k: v for k, v in sorted(groupby_list.items(), key=lambda item: len(item[1]), reverse=True)}
    for k, v in groupby_list.items():
        print(k, len(v))
        merging_samples = cluster_df_orig[cluster_df_orig['remapped_label'].isin(v)]
        merging_sample_paths = merging_samples['global_path'].to_list()
        images = []
        for path in tqdm(merging_sample_paths, desc=f'{k}'):
            if path in img_reader.paths:
                img = img_reader.read_by_path(path)
                images.append(img)
            if len(images) == 48:
                break
        print('images', len(images))

        from general_utils.img_utils import stack_images
        # convert numpy image to uint8
        nrows = len(images) // 8
        images = [image[0].astype(np.uint8) for image in images]
        stacked_img = stack_images(images, num_rows=nrows, num_cols=8)
        cv2.imwrite(f'/mckim/temp/{k}_len{len(v)}.png', stacked_img[:,:,::-1])



    # # see how labels change
    # def remap(x):
    #     return x if x not in label_mapping else label_mapping[x]
    # cluster_df['remapped_label'] = cluster_df['remapped_label'].apply(remap)
    # cluster_df = cluster_df[cluster_df['remapped_label'] != -1]
    # cluster_df.loc[:, 'count'] = cluster_df['remapped_label'].groupby(cluster_df['remapped_label']).transform('count')
    # cluster_df = cluster_df[cluster_df['count'] >= 2]
    # n_subjects = len(cluster_df['remapped_label'].unique())
    # n_images = len(cluster_df)


    # skipped_components = torch.load(os.path.join(root, 'skipped_components.pth'))


    

if __name__ == '__main__':
    main()