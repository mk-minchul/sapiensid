from datasets import Dataset
import torch
from functools import partial
from tqdm import tqdm
import os
import numpy as np
try:
    from .base_evaluator import BaseEvaluator
    from .body_ccvid.evaluate import evaluate
except:
    from base_evaluator import BaseEvaluator
    from body_ccvid.evaluate import evaluate

def preprocess_transform(examples, image_transforms):
    images = [image.convert("RGB") for image in examples['image']]
    images = [image_transforms(image) for image in images]
    examples["pixel_values"] = images
    return examples


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    indexes = torch.tensor([example["index"] for example in examples], dtype=torch.int)
    image_paths = [example["path"] for example in examples]
    pids = torch.tensor([example["pid"] for example in examples], dtype=torch.int)
    camids = torch.tensor([example["camid"] for example in examples], dtype=torch.int)
    clothes_ids = torch.tensor([example["clothes_id"] for example in examples], dtype=torch.int)

    return {
        "pixel_values": pixel_values,
        "index": indexes,
        "image_paths": image_paths,
        "pid": pids,
        "camid": camids,
        "clothes_id": clothes_ids
    }


class BodyCCVIDEvaluator(BaseEvaluator):
    def __init__(self, name, data_path, transform, fabric, batch_size, num_workers):
        super().__init__(name, fabric, batch_size)
        self.name = name
        self.data_path = data_path
        dataset = Dataset.load_from_disk(data_path)
        preprocess = partial(preprocess_transform, image_transforms=transform)
        dataset = dataset.with_transform(preprocess)
        self.dataloader = fabric.setup_dataloader_from_dataset(dataset,
                                                               is_train=False,
                                                               batch_size=batch_size,
                                                               num_workers=num_workers,
                                                               collate_fn=collate_fn)
        self.meta = torch.load(os.path.join(data_path, 'metadata.pt'))

        self.image_level_eval = True



    def integrity_check(self, eval_color_space, pipeline_color_space):
        assert eval_color_space == pipeline_color_space


    @torch.no_grad()
    def evaluate(self, pipeline, epoch=0, step=0, n_images_seen=0):
        pipeline.eval()
        collection = self.extract(pipeline)
        collection_flip = self.extract(pipeline, flip_images=True)
        if self.fabric.local_rank == 0:
            result = self.compute_metric(collection, collection_flip)
            self.log(result, epoch, step, n_images_seen)
        else:
            result = {}
        return result

    def extract(self, pipeline, flip_images=False):
        all_features = []
        all_index = []
        all_image_paths = []
        all_pids = []
        all_camids = []
        all_clothes_ids = []
        for batch_idx, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc='Extracting Feature',
                                     disable=self.fabric.local_rank != 0):
            batch = self.complete_batch(batch)  # needed for last batch to be gather compatible

            if self.is_debug_run():
                if batch_idx > 10:
                    break

            images = batch['pixel_values']
            index = batch['index']
            image_paths = batch['image_paths']

            if flip_images:
                images = torch.flip(images, dims=[3])
            features = pipeline(images)
            all_features.append(features.cpu().detach())
            all_index.append(index.cpu().detach())
            all_image_paths.extend(image_paths)
            all_pids.append(batch['pid'].cpu().detach())
            all_camids.append(batch['camid'].cpu().detach())
            all_clothes_ids.append(batch['clothes_id'].cpu().detach())

        # aggregate across all gpus
        per_gpu_collection = {"index": torch.cat(all_index, dim=0),
                              'features': torch.cat(all_features, dim=0),
                              'image_paths': all_image_paths,
                              'pid': torch.cat(all_pids, dim=0),
                              'camid': torch.cat(all_camids, dim=0),
                              'clothes_id': torch.cat(all_clothes_ids, dim=0)}

        # cpu based gathering just in case we have a lot of data
        collection = self.gather_collection(method='cpu', per_gpu_collection=per_gpu_collection)
        return collection


    def compute_metric(self, collection, collection_flip):
        if self.is_debug_run():
            print('Debug run, skipping metric computation')
            ranks = [1, 5, 20]
            return {k: 0.0 for k in ['rank-{}'.format(r) for r in ranks]}

        embeddings = (collection['features'] + collection_flip['features'])
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        embeddings = embeddings.cpu().detach()
        image_paths = collection['image_paths']
        pids = collection['pid'].cpu().detach()
        camids = collection['camid'].cpu().detach()
        clothes_ids = collection['clothes_id'].cpu().detach()

        if self.image_level_eval:
            result = evaluate(
                embeddings=embeddings,
                image_paths=image_paths,
                pids=pids,
                camids=camids,
                clothes_ids=clothes_ids,
                meta=self.meta,
            )
        else:
            track_ids = ["/".join(path.split('/')[-3:-1]) for path in image_paths]
            track_id_to_index = {}
            for i, track_id in enumerate(track_ids):
                if track_id not in track_id_to_index:
                    track_id_to_index[track_id] = []
                track_id_to_index[track_id].append(i)
            
            agg_embeddings = []
            agg_pids = []
            agg_camids = []
            agg_clothes_ids = []
            agg_image_paths = []
            meta_data_label = []
            full_meta_data_label = np.array(self.meta['full_meta_data_label'])
            for track_id, index in track_id_to_index.items():
                agg_embeddings.append(embeddings[index].mean(dim=0))
                assert pids[index].unique().shape[0] == 1
                assert camids[index].unique().shape[0] == 1
                assert clothes_ids[index].unique().shape[0] == 1
                agg_pids.append(pids[index][0])
                agg_camids.append(camids[index][0])
                agg_clothes_ids.append(clothes_ids[index][0])
                agg_image_paths.extend(image_paths[index][0])
                meta_data_label.append(full_meta_data_label[index][0])
            agg_embeddings = torch.stack(agg_embeddings)
            agg_embeddings = agg_embeddings / agg_embeddings.norm(dim=1, keepdim=True)

            # adjust meta info as well
            self.meta['full_meta_data_label'] = meta_data_label
            agg_pids = torch.tensor(agg_pids)
            agg_camids = torch.tensor(agg_camids)
            agg_clothes_ids = torch.tensor(agg_clothes_ids)

            result = evaluate(
                embeddings=agg_embeddings,
                image_paths=agg_image_paths,
                pids=agg_pids,
                camids=agg_camids,
                clothes_ids=agg_clothes_ids,
                meta=self.meta,
            )
        return result


if __name__ == '__main__':
    evaluator = BodyCCVIDEvaluator(name='body_ccvid_evaluator', data_path='/ssd2/data/body/validation_sets/prcc_test', transform=None, fabric=None, batch_size=128, num_workers=0)
    evaluator.evaluate(pipeline=None, epoch=0, step=0, n_images_seen=0)