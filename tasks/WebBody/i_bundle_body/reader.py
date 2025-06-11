from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import os
import pandas as pd
import mxnet as mx
import cv2
import atexit
import numpy as np
import torch


class RecordReader():

    def __init__(self, root='/mckim/temp/temp_recfiles', prefix='train', image_size=384):
        path_imgidx = os.path.join(root, f'{prefix}.idx')
        path_imgrec = os.path.join(root, f'{prefix}.rec')
        self.record = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

        path_list = os.path.join(root, f'{prefix}.tsv')
        info = pd.read_csv(path_list, sep='\t', index_col=0, header=None)
        self.index_to_path = dict(info[1])
        self.path_to_index = {v:k for k,v in self.index_to_path.items()}
        # self.info = info
        # self.info.columns = ['path', 'label']
        del info # free memory
        atexit.register(self.dispose)

        self.body_kps_xyn = pd.read_csv(os.path.join(root, f'body_kps_xyn.csv'))
        self.bbox_xyxyn = pd.read_csv(os.path.join(root, f'bbox_xyxyn.csv'))
        self.keypoint_cols = np.array([[f'kp_{k}_x', f'kp_{k}_y'] for k in range(17)]).flatten()
        self.bbox_cols = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']
        self.resize_side = image_size


    def dispose(self):
        self.record.close()

    def read_by_index(self, index):
        header, binary = mx.recordio.unpack(self.record.read_idx(index))
        image = mx.image.imdecode(binary).asnumpy()
        path = self.index_to_path[index]
        kps = self.load_keypoints_for_crop(index)

        # 1. Resize the image to max side 384 (keep aspect ratio)
        h, w = image.shape[:2]
        scale = self.resize_side / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h))

        # 2. Zero pad the image to self.resize_sidexself.resize_side
        pad_h, pad_w = self.resize_side - new_h, self.resize_side - new_w
        top_pad, left_pad = pad_h // 2, pad_w // 2
        bottom_pad, right_pad = pad_h - top_pad, pad_w - left_pad
        image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # 3. Adjust the keypoints to accommodate the padding
        invisible_idx = kps == -1
        kps[:, 0] = (kps[:, 0] * new_w + left_pad) / self.resize_side
        kps[:, 1] = (kps[:, 1] * new_h + top_pad) / self.resize_side
        kps[invisible_idx] = -1
        return image, path, kps, [h,w]
    
    def load_keypoints_for_crop(self, index):
        orig_kps = torch.from_numpy(self.body_kps_xyn.loc[index, self.keypoint_cols].to_numpy())
        orig_bboxes = torch.from_numpy(self.bbox_xyxyn.loc[index, self.bbox_cols].to_numpy())
        orig_kps = orig_kps.reshape(1, -1, 2)
        orig_bboxes = orig_bboxes.reshape(1, 1, 4)
        crop_wh = orig_bboxes[:, :, 2:] - orig_bboxes[:, :, :2]
        crop_kps = (orig_kps - orig_bboxes[:, :, :2]) / crop_wh
        crop_kps[orig_kps == 0] = -1  # Set invisible keypoints to -1
        return crop_kps[0]

    def read_by_path(self, path):
        index = self.path_to_index[path]
        return self.read_by_index(index)



class SplittedRecordReader():
    def __init__(self, roots, image_size=384):
        print(f'Loading {len(roots)} records')
        print(roots)
        self.records = []
        for root in tqdm(roots, total=len(roots), desc='Loading records'):
            self.records.append(RecordReader(root, image_size=image_size))
        self.path_to_record_num = {}
        for record_idx, record in enumerate(self.records):
            for key in record.path_to_index.keys():
                self.path_to_record_num[key] = record_idx
        self.paths = list(self.path_to_record_num.keys())

    def __len__(self):
        return len(self.paths)

    def read_by_index(self, index):
        return self.read_by_path(self.paths[index])

    def read_by_path(self, path):
        record_num = self.path_to_record_num[path]
        return self.records[record_num].read_by_path(path)

    def export(self, save_root):
        raise NotImplementedError('')

    def existing_keys(self):
        return self.path_to_record_num.keys()

    def load_done_list(self):
        donelist = set()
        for record in self.records:
            _donelist = record.load_done_list()
            if _donelist is not None:
                donelist = donelist | _donelist
        return donelist

