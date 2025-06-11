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

    def __init__(self, root='/mckim/temp/temp_recfiles', prefix='train'):
        path_imgidx = os.path.join(root, f'{prefix}.idx')
        path_imgrec = os.path.join(root, f'{prefix}.rec')
        self.record = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

        path_list = os.path.join(root, f'{prefix}.tsv')
        info = pd.read_csv(path_list, sep='\t', index_col=0, header=None)
        self.index_to_path = dict(info[1])
        self.path_to_index = {v:k for k,v in self.index_to_path.items()}
        self.info = info
        self.info.columns = ['path', 'label']
        atexit.register(self.dispose)

        if os.path.exists(os.path.join(root, f'shapes.csv')):
            self.shapes = pd.read_csv(os.path.join(root, f'shapes.csv'))
            self.shapes = self.shapes[['height', 'width']]
        else:
            self.shapes =  None

        if os.path.exists(os.path.join(root, f'kps_xyn.csv')):
            self.kps_xyn = pd.read_csv(os.path.join(root, f'kps_xyn.csv'))
            kp_header = np.array([[f'kp_{i}_x' for i in range(17)], [f'kp_{i}_y' for i in range(17)]]).T.flatten().tolist()
            self.kps_xyn = self.kps_xyn[kp_header]
        else:
            self.kps_xyn = pd.read_csv(os.path.join(root, f'face_kps_xyn.csv'))
            kp_header = np.array([[f'kp_{i}_x' for i in range(5)], [f'kp_{i}_y' for i in range(5)]]).T.flatten().tolist()
            self.kps_xyn = self.kps_xyn[kp_header]


    def dispose(self):
        self.record.close()

    def read_by_index(self, index):
        header, binary = mx.recordio.unpack(self.record.read_idx(index))
        label = header.label
        if index % 1000 == 0:
            assert label == self.info.loc[index, 'label']
        image = mx.image.imdecode(binary).asnumpy()
        kps = self.kps_xyn.loc[index].to_numpy()
        if self.shapes is not None:
            shape = self.shapes.loc[index].to_numpy()
        else:
            shape = None
        path = self.index_to_path[index]
        return image, kps, shape, path, label
    

    def read_by_path(self, path):
        index = self.path_to_index[path]
        return self.read_by_index(index)



