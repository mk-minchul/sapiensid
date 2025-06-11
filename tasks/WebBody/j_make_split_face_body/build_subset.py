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
import pandas as pd
from PIL import Image
from reader import RecordReader
from writer import Writer, ArrayWriter
from tqdm import tqdm

def main(args):
    reader = RecordReader(root=args.data_root, prefix='train')
    save_dir = os.path.join(args.data_root + f'_{args.config_path.split("/")[-1].split(".")[0]}')

    subset_config = pd.read_csv(os.path.join(args.data_root, args.config_path), sep='\t', header=None, index_col=0)
    subset_config.columns = ['path', 'label']
    label_mapping = {label: i for i, label in enumerate(sorted(subset_config['label'].unique()))}

    idx_header = ['local_idx', 'global_idx', 'detection_idx']
    kp_header = np.array([[f'kp_{i}_x' for i in range(17)], [f'kp_{i}_y' for i in range(17)]]).T.flatten().tolist()
    kp_writer = ArrayWriter(save_dir, 'kps_xyn.csv', idx_header, kp_header, n_digits=5)
    shape_writer = ArrayWriter(save_dir, 'shapes.csv', idx_header, ['height', 'width'], n_digits=0)
    writer = Writer(save_dir, prefix='train')

    subset_indices = subset_config.index
    n = 0
    for i in tqdm(subset_indices, desc='Building subset', total=len(subset_indices), ncols=100):
        image, kps, shape, path, label = reader.read_by_index(i)
        rgb_pil_img = Image.fromarray(image)
        mapped_label = label_mapping[int(label)]

        global_index, det_index = path.split('/')[-1].replace('.jpg', '').split('_')
        global_index, det_index = int(global_index), int(det_index)
        kp_writer.write_array(n, kps, global_idx=global_index, detection_idx=det_index)
        shape_writer.write_array(n, shape, global_idx=global_index, detection_idx=det_index)
        writer.write(rgb_pil_img=rgb_pil_img, save_path=path, label=mapped_label, bgr=False)
        writer.mark_done(n, path)
        n += 1

        if i % args.vis_every == 0 or i < 50:
            os.makedirs(f'{save_dir}/vis', exist_ok=True)
            rgb_pil_img.save(f'{save_dir}/vis/{i}_{mapped_label}.png')

    kp_writer.close()
    shape_writer.close()
    writer.close()

    with open(os.path.join(save_dir, 'done.txt'), 'w') as f:
        f.write('done')
    with open(os.path.join(save_dir, 'statistics.txt'), 'w') as f:
        info = pd.read_csv(os.path.join(save_dir, 'train.tsv'), sep='\t', header=None)
        info.columns = ['_idx', 'path', 'label']
        f.write(f'num_images: {len(info)}\n')
        f.write(f'num_labels: {len(info["label"].unique())}\n')
        f.write(f'num_images_per_label: {info["label"].value_counts().mean()}\n')
        f.write(f'smallest_label: {info["label"].min()}\n')
        f.write(f'largest_label: {info["label"].max()}\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/hdd4/data/faces/webface260m/WebFaceV2_subset_by_face_score_v1/whole_body_bundle/raw_img_parquets_640_yolo_cropped/cluster_eps:0.7_min_samples:2_max_group_per_label:1/facefilter_0.95_merge_threshold_0.7_ambiguous_threshold_0.6_val_sim_threshold_0.7_image_size_384')
    parser.add_argument('--config_path', type=str, default='subset_config/subset_4m.tsv')
    parser.add_argument('--vis_every', type=int, default=100000)
    args = parser.parse_args()
    main(args)