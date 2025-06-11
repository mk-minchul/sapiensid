from torch.utils.data import Dataset, DataLoader
from PIL import Image

class RecordDatasetFromList(Dataset):
    def __init__(self, reader, transform, img_list=None):
        super(RecordDatasetFromList, self).__init__()
        if img_list is None:
            img_list = reader.paths
        self.img_list = img_list
        self.reader = reader
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        try:
            result = self.reader.read_by_path(self.img_list[idx])
            if len(result) == 2:
                img, path = result
                crop_kp = None
            else:
                img, path, crop_kp = result

            if img is None:
                raise ValueError(f"Image at path {self.img_list[idx]} could not be loaded.")

            img = img[:, :, :3]  # Ensure image has 3 channels
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, idx, path, crop_kp

        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            return -1, idx, -1, -1
