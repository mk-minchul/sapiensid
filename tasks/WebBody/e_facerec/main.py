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
HF_TOKEN = os.getenv('HF_TOKEN')
assert HF_TOKEN is not None, 'HF_TOKEN is not set'

import torch
import argparse
from model_helper import load_model
from dataset_helper import make_dataset
from writer import FeatureWriter, ArrayWriter
from general_utils.huggingface_model_utils import load_model_by_repo_id

from tqdm import tqdm
import lovely_tensors as lt
lt.monkey_patch()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='do test')
    parser.add_argument('--data_root', type=str, default='/hdd3/data/faces/webface260m/WebFaceV2_subset_by_face_score_v1/raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320')
    parser.add_argument('--parquet_num_start', type=int, default=3)
    parser.add_argument('--parquet_num_end', type=int, default=4)
    parser.add_argument('--save_root', default='/ssd2/data/faces/webface260m/WebFaceV2_subset_by_face_score_v1')
    parser.add_argument('--task', default='foundface_007')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mock', action='store_true')

    torch.set_float32_matmul_precision('high')

    # set args
    args = parser.parse_args()
    args.num_workers = 0 if args.mock else args.num_workers

    # make save path
    model_name = 'vit_base_kprpe_webface12m'
    parquet_name = f'parquet_{args.parquet_num_start}'
    if args.parquet_num_end != -1:
        assert args.parquet_num_end >= args.parquet_num_start
        parquet_name += f'-{args.parquet_num_end}'
    save_path = os.path.join(args.save_root,
                             'face_features',
                             model_name,
                             'raw_img_parquets_640_yolo_cropped_private_retinaface_resnet50_aligned_cropsize_160_maxsize_320',
                             parquet_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print('save_path', save_path)

    # load model
    path = os.path.expanduser('~/.cvlface_cache/minchul/cvlface_adaface_vit_base_kprpe_webface12m')
    repo_id = 'minchul/cvlface_adaface_vit_base_kprpe_webface12m'
    model = load_model_by_repo_id(repo_id, path, HF_TOKEN)
    model.to('cuda')

    # make dataset
    dataloader = make_dataset(args.data_root,
                              args.parquet_num_start, args.parquet_num_end,
                              args.batch_size, args.num_workers)

    prev_max_indices = -1
    n = 0
    writer = FeatureWriter(save_path, prefix='train')
    idx_header = ['local_idx', 'global_idx', 'detection_idx']
    norm_writer = ArrayWriter(save_path, 'feature_norm.csv', idx_header, ['norm'], n_digits=2)
    label_writer = ArrayWriter(save_path, 'label.csv', idx_header, ['label'], n_digits=0)


    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Making {parquet_name}', ncols=75, miniters=10):
        x, indices, paths, crop_kps = batch
        if x.ndim == 1:
            print('Error during batch making')
            continue

        try:
            max_indices = max(indices).item()
            assert max_indices > prev_max_indices
            prev_max_indices = max_indices

            x = x.cuda(non_blocking=True)
            crop_kps = crop_kps.cuda(non_blocking=True)
            # from general_utils.img_utils import visualize
            # visualize(x.cpu(), crop_kps.cpu()).save('/mckim/temp/temp.png')

            global_indices = [int(path.split('/')[1].split('_')[0]) for path in paths]
            labels = [int(path.split('/')[0]) for path in paths]
            det_indices = [int(path.split('/')[1].split('_')[1].split('.')[0]) for path in paths]
            features = model(x, crop_kps)
            norms = torch.norm(features, dim=1)

            for b_idx in range(len(features)):
                global_index = global_indices[b_idx]
                det_index = det_indices[b_idx]
                feature = features[b_idx]
                norm = norms[b_idx]
                label = labels[b_idx]
                feature_save_path = paths[b_idx]
                save_features_np = feature.detach().cpu().numpy()
                save_features_np_fp16 = save_features_np.astype(np.float16)
                writer.write(save_features_np_fp16, feature_save_path, label)
                norm_writer.write_array(n, np.array([norm.item()]), global_idx=global_index, detection_idx=det_index)
                label_writer.write_array(n, np.array([label]), global_idx=global_index, detection_idx=det_index)
                writer.mark_done(n, feature_save_path)
                n += 1

        except Exception as e:
            print(f'Error during parsing info: {e}')
            continue

    # save a done.txt file
    with open(os.path.join(save_path, 'done.txt'), 'w') as f:
        f.write('done\n')

    # close files
    norm_writer.close()
    label_writer.close()
    writer.close()