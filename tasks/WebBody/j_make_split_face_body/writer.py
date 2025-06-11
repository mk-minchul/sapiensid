
import os
import mxnet as mx
try:
    import Queue as queue
except ImportError:
    import queue
import numpy as np
import atexit

def prepare_record_saver(save_root, prefix='train'):

    os.makedirs(save_root, exist_ok=True)
    q_out = queue.Queue()
    fname_rec = f'{prefix}.rec'
    fname_idx = f'{prefix}.idx'
    fname_list = f'{prefix}.tsv'
    done_list = 'done_list.txt'

    if os.path.isfile(os.path.join(save_root, fname_idx)):
        os.remove(os.path.join(save_root, fname_idx))
    if os.path.isfile(os.path.join(save_root, fname_rec)):
        os.remove(os.path.join(save_root, fname_rec))
    if os.path.isfile(os.path.join(save_root, fname_list)):
        os.remove(os.path.join(save_root, fname_list))
    if os.path.isfile(os.path.join(save_root, done_list)):
        os.remove(os.path.join(save_root, done_list))

    record = mx.recordio.MXIndexedRecordIO(os.path.join(save_root, fname_idx),
                                           os.path.join(save_root, fname_rec), 'w')
    list_writer = open(os.path.join(save_root, fname_list), 'w')
    mark_done_writer = open(os.path.join(save_root, done_list), 'w')

    return record, q_out, list_writer, mark_done_writer


class Writer():

    def __init__(self, save_root, prefix='train', write_landmarks=False):
        record, q_out, list_writer, mark_done_writer = prepare_record_saver(save_root, prefix=prefix)
        self.record = record
        self.list_writer = list_writer
        self.mark_done_writer = mark_done_writer  # needed for continuing
        self.q_out = q_out
        self.image_index = 0
        if write_landmarks:
            self.ldmk_writer = open(os.path.join(save_root, 'ldmk_5points.csv'), 'w')
            self.ldmk_writer.write('idx,ldmk_0,ldmk_1,ldmk_2,ldmk_3,ldmk_4,ldmk_5,ldmk_6,ldmk_7,ldmk_8,ldmk_9\n')


        atexit.register(self.record.close)

    def write(self, rgb_pil_img, save_path, label, bgr=False, ldmks=None):
        assert isinstance(label, int)
        header = mx.recordio.IRHeader(0, label, self.image_index, 0)
        if bgr:
            # this saves in bgr
            s = mx.recordio.pack_img(header, np.array(rgb_pil_img), quality=95, img_fmt='.jpg')
        else:
            # this saves in rgb
            s = mx.recordio.pack_img(header, np.array(rgb_pil_img)[:,:,::-1], quality=95, img_fmt='.jpg')
        item = [self.image_index, save_path, label]

        self.q_out.put((item[0], s, item))
        _, s, _ = self.q_out.get()
        self.record.write_idx(item[0], s)
        line = f'{self.image_index}\t{save_path}\t{label}\n'
        self.list_writer.write(line)
        if ldmks is not None:
            self.write_landmarks(ldmks, self.image_index)

        self.image_index = self.image_index + 1

    def write_landmarks(self, ldmks, idx):
        assert hasattr(self, 'ldmk_writer')
        assert ldmks.ndim == 2
        assert ldmks.shape[0] == 5
        assert ldmks.shape[1] == 2
        ldmk = ldmks.reshape(-1)
        self.ldmk_writer.write(f'{idx},{ldmk[0]:.5f},'
                               f'{ldmk[1]:.5f},'
                               f'{ldmk[2]:.5f},'
                               f'{ldmk[3]:.5f},'
                               f'{ldmk[4]:.5f},'
                               f'{ldmk[5]:.5f},'
                               f'{ldmk[6]:.5f},'
                               f'{ldmk[7]:.5f},'
                               f'{ldmk[8]:.5f},'
                               f'{ldmk[9]:.5f}\n')

    def close(self):
        self.record.close()
        self.list_writer.close()
        self.mark_done_writer.close()
        if hasattr(self, 'ldmk_writer'):
            self.ldmk_writer.close()

    def mark_done(self, context, name):
        line = '%s\t' % context + '%s\n' % name
        self.mark_done_writer.write(line)



class ArrayWriter:
    def __init__(self, save_root, filename, idx_header, array_header, n_digits=5):
        """
        Initializes the ArrayWriter class.

        :param save_root: Directory where the CSV file will be saved.
        :param filename: Name of the CSV file.
        :param array_header: List of column names for the array_header.
        """
        os.makedirs(save_root, exist_ok=True)
        self.filepath = os.path.join(save_root, filename)
        self.idx_header = idx_header
        self.array_header = array_header
        self.header = idx_header + array_header
        assert self.header[0] == 'local_idx'

        self.use_global_index = len(self.header) > 1
        if self.use_global_index:
            assert self.header[1] == 'global_idx'
        self.use_detection_index = len(self.header) > 2
        if self.use_detection_index:
            assert self.header[2] == 'detection_idx'

        self._initialize_writer()
        self.n_digits = n_digits

    def _initialize_writer(self):
        """
        Initializes the file writer and writes the header.
        """
        self.array_writer = open(self.filepath, 'w')
        header_line = ','.join(self.header) + '\n'
        self.array_writer.write(header_line)

    def write_array(self, idx, array, global_idx=None, detection_idx=None):
        """
        Flattens the array and writes it to the file along with the index.

        :param idx: The index to write before the array.
        :param array: The NumPy array to be flattened and written.
        """
        flattened_array = array.flatten()
        formatted_values = ','.join(f'{value:.{self.n_digits}f}' if value != 0 else '0' for value in flattened_array)
        fstring = f'{idx}'
        if self.use_global_index:
            assert global_idx is not None
            fstring = fstring + f',{global_idx}'
        if self.use_detection_index:
            assert detection_idx is not None
            fstring = fstring + f',{detection_idx}'
        fstring = fstring + f',{formatted_values}\n'
        self.array_writer.write(fstring)

    def close(self):
        """
        Closes the file writer.
        """
        self.array_writer.close()