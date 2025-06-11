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
import os
import argparse
from reader import SplittedRecordReader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torch.utils.data import Dataset, DataLoader
from general_utils.huggingface_model_utils import load_model_by_repo_id
from tqdm import tqdm
from dataset_helper import RecordDatasetFromList
from alignment_helper import align_larger_crop, pad_to_square, crop_upper_body_from_body
from general_utils.img_utils import tensor_to_pil, concat_pil, stack_images, visualize
from visual_helper import visualize_body
import lovely_tensors as lt
lt.monkey_patch()
from writer import Writer, ArrayWriter



if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/ssd2/data/faces/webface260m/WebFaceV2_subset_by_face_score_v1/raw_img_parquets_640_yolo_cropped")
    parser.add_argument("--parquet_num", type=str, default="3")
    parser.add_argument("--save_root", type=str, default='/ssd2/data/faces/webface260m/WebFaceV2_subset_by_face_score_v1/')
    parser.add_argument("--aligner_repo_id", type=str, default="minchul/private_retinaface_resnet50")
    parser.add_argument("--crop_size", type=int, default=160)
    parser.add_argument("--max_save_size", type=int, default=320)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--vis_every", type=int, default=10000)
    parser.add_argument("--debug", action='store_true')

    args = parser.parse_args()

    paths = get_all_files(args.root_dir, extension_list=['.rec'], sort=True)
    print('Found records:', len(paths))
    in_text = f'{args.parquet_num}_parquet'
    paths = [os.path.dirname(path) for path in paths if in_text in path]
    print('Will Process records:', len(paths))

    # load aligner
    HF_TOKEN = os.environ['HF_TOKEN']
    assert HF_TOKEN
    aligner = load_model_by_repo_id(repo_id=args.aligner_repo_id,
                                    save_path=os.path.expanduser(f'~/cache/{args.aligner_repo_id}'),
                                    HF_TOKEN=HF_TOKEN).to('cuda').eval()
    print(f'Aligner {args.aligner_repo_id} loaded')

    # make dataset
    reader = SplittedRecordReader(paths)
    transform = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])
    dataset = RecordDatasetFromList(reader, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # make save_path
    aligner_name = os.path.basename(args.aligner_repo_id)
    save_dir = os.path.join(args.save_root, os.path.basename(args.root_dir)+f'_{aligner_name}_aligned_cropsize_{args.crop_size}_maxsize_{args.max_save_size}')
    save_dir = os.path.join(save_dir, in_text)

    # make writer
    idx_header = ['local_idx', 'global_idx', 'detection_idx']
    face_kp_header = np.array([[f'kp_{i}_x' for i in range(5)], [f'kp_{i}_y' for i in range(5)]]).T.flatten().tolist()
    confidence_header = ['confidence']
    face_kp_writer = ArrayWriter(save_dir, 'face_kps_xyn.csv', idx_header, face_kp_header, n_digits=5)
    confidence_writer = ArrayWriter(save_dir, 'face_confidence.csv', idx_header, confidence_header, n_digits=5)
    writer = Writer(save_dir, prefix='train')
    vis_dir = os.path.join(save_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    prev_max_indices = -1
    n = 0
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Making {in_text}', ncols=70, miniters=10):
        x, indices, paths, crop_kps = batch
        if x.ndim == 1:
            # error during batch making
            print('Error during batch making')
            continue

        try:
            # parse info
            assert len(paths) == 1, 'batch size should be 1'
            path = paths[0]
            label = int(path.split('/')[0])
            global_index = int(path.split('/')[1].split('_')[0])
            det_index = int(path.split('/')[1].split('_')[1].split('.')[0])

            # shuffle check
            max_indices = max(indices).item()
            assert max_indices > prev_max_indices
            prev_max_indices = max_indices

            # to cuda
            x = x.to('cuda')
            crop_kps = crop_kps.to('cuda')

            # precrop to face region based on yolo points
            cropped_x, bbox_xyxyn = crop_upper_body_from_body(x, crop_kps)
            cropped_x_padded = pad_to_square(cropped_x, value=-1)
            side = cropped_x_padded.shape[-1]

            # align using model
            aligned_x, orig_pred_ldmks, aligned_ldmks, scores, thetas, bbox = aligner(cropped_x_padded, padding_ratio_override=None)
            bbox = bbox[0].view(2, 2)
            bbox_max_side = ((bbox[1] - bbox[0]).max() * 2 * side).long().item()
            if bbox_max_side < 16:
                # too small bbox, skip
                continue

            aligned_large_crop_x, large_crop_ldmks = align_larger_crop(thetas, args.crop_size, min(args.max_save_size, bbox_max_side), cropped_x_padded, orig_pred_ldmks)

            kp = large_crop_ldmks[0]
            conf = scores[0]
            rgb_pil_img = tensor_to_pil(aligned_large_crop_x)[0]
            face_kp_writer.write_array(n, kp.cpu().numpy(), global_idx=global_index, detection_idx=det_index)
            confidence_writer.write_array(n, np.array([conf.item()]), global_idx=global_index, detection_idx=det_index)
            writer.write(rgb_pil_img=rgb_pil_img, save_path=path, label=label, bgr=False)
            writer.mark_done(n, path)
            if n % args.vis_every == 0:
                rgb_pil_img.save(f'{vis_dir}/{n}.png')
            n += 1

            if args.debug:
                if n > 100:
                    break
                os.makedirs('/mckim/temp/body_crop_vis', exist_ok=True)
                os.makedirs('/mckim/temp/body_crop_vis_only_face', exist_ok=True)
                visualize_body(x.cpu(), crop_kps, bbox_xyxyn).save(f'/mckim/temp/body_crop_vis/raw_{idx}.png')
                visualize_body(cropped_x.cpu()).save(f'/mckim/temp/body_crop_vis/cropped_{idx}.png')
                visualize_body(cropped_x_padded.cpu()).save(f'/mckim/temp/body_crop_vis/cropped_padded_{idx}.png')
                # visualize_body(cropped_x_padded.cpu(), orig_pred_ldmks, bbox).save(f'/mckim/temp/body_crop_vis/cropped_padded_bbox_{idx}.png')

                visualize(aligned_x.cpu(), aligned_ldmks).save(f'/mckim/temp/body_crop_vis/aligned_vis_{idx}.png')
                visualize(aligned_large_crop_x.cpu(), large_crop_ldmks).save(f'/mckim/temp/body_crop_vis/aligned_large_vis_{idx}.png')
                visualize(aligned_large_crop_x.cpu(), large_crop_ldmks).save(f'/mckim/temp/body_crop_vis_only_face/aligned_large_vis_{idx}.png')
                save_side = min(args.max_save_size, bbox_max_side)
                margin = int((args.crop_size - 112) // 2 / args.crop_size * save_side)
                concat_pil(tensor_to_pil(aligned_large_crop_x[:, :, margin:-margin, margin:-margin])).save(f'/mckim/temp/body_crop_vis/aligned_center_{idx}.png')

        except Exception as e:
            print(f'Error during loop processing {idx}: {e}')
            continue

    # save a done.txt file
    with open(os.path.join(save_dir, 'done.txt'), 'w') as f:
        f.write('done\n')

    # close files
    face_kp_writer.close()
    confidence_writer.close()
    writer.close()
