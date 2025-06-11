import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class SimilarityModel(torch.nn.Module):

    def __init__(self, similarity_threshold, ambiguous_threshold=None, features=None, labels=None, counts=None, indexes=None, split_size=24576):
        super(SimilarityModel, self).__init__()
        self.similarity_threshold = similarity_threshold
        self.ambiguous_threshold = ambiguous_threshold
        self.splitted_features = torch.split(features, split_size)
        self.splitted_indexes = torch.split(indexes, split_size)
        self.splitted_labels = torch.split(labels, split_size)
        self.splitted_counts = torch.split(counts, split_size)

    def forward(self, features, labels, counts, indexes):
        similar_pairs = []
        ambiguous_pairs = []
        for feature2, labels2, counts2, indexes2 in zip(self.splitted_features, self.splitted_labels, self.splitted_counts, self.splitted_indexes):
            features2 = feature2.to(features.device)
            similar_pairs_batch, ambiguous_pairs_batch = self._forward(features, labels, counts, indexes, features2, labels2, counts2, indexes2)
            similar_pairs.extend(similar_pairs_batch.tolist())
            ambiguous_pairs.extend(ambiguous_pairs_batch.tolist())
        similar_pairs = torch.tensor(similar_pairs).to(features.device)
        ambiguous_pairs = torch.tensor(ambiguous_pairs).to(features.device)
        if similar_pairs.ndim == 1:
            similar_pairs = similar_pairs.view(-1, 2)
        if ambiguous_pairs.ndim == 1:
            ambiguous_pairs = ambiguous_pairs.view(-1, 2)
        return similar_pairs, ambiguous_pairs

    def _forward(self, features1, labels1, counts1, indexes1, features2, labels2, counts2, indexes2):
        features1_norm = F.normalize(features1, p=2, dim=1)
        features2_norm = F.normalize(features2, p=2, dim=1)
        similarity = torch.mm(features1_norm, features2_norm.t())
        labels1 = labels1.to(similarity.device)
        labels2 = labels2.to(similarity.device)
        indexes1 = indexes1.to(similarity.device)
        indexes2 = indexes2.to(similarity.device)
        counts1 = counts1.to(similarity.device)
        counts2 = counts2.to(similarity.device)
    
        # Find pairs above threshold
        condition = similarity > self.similarity_threshold
        row_indices, col_indices = torch.nonzero(condition, as_tuple=True)
        similar_pairs, _ = self.get_condition_pairs(row_indices, col_indices, labels1, labels2, indexes1, indexes2)

        if self.ambiguous_threshold is not None:
            condition = (similarity > self.ambiguous_threshold) & (similarity < self.similarity_threshold)
            row_indices, col_indices = torch.nonzero(condition, as_tuple=True)
            ambiguous_pairs, valid_pairs_mask = self.get_condition_pairs(row_indices, col_indices, labels1, labels2, indexes1, indexes2)

            counts_pairs = torch.stack([
                counts1[row_indices[valid_pairs_mask]],
                counts2[col_indices[valid_pairs_mask]]
            ], dim=1)

            # sort such that the smaller count is first between each pair
            sorted_indices = torch.argsort(counts_pairs, dim=1)
            ambiguous_pairs = torch.gather(ambiguous_pairs, 1, sorted_indices)
        else:
            ambiguous_pairs = torch.tensor([], dtype=torch.long)

        return similar_pairs, ambiguous_pairs

    def get_condition_pairs(self, row_indices, col_indices, labels1, labels2, indexes1, indexes2):
        # Compute global indices
        global_row_indices = indexes1[row_indices]
        global_col_indices = indexes2[col_indices]

        # Create a mask for valid pairs (avoid self-comparisons, duplicates, and same labels)
        no_self_comparison_mask = global_row_indices < global_col_indices
        diff_label_mask = labels1[row_indices] != labels2[col_indices]
        valid_pairs_mask = no_self_comparison_mask & diff_label_mask

        # Apply the mask and add to similar_pairs
        label_pairs = torch.stack([
            labels1[row_indices[valid_pairs_mask]],
            labels2[col_indices[valid_pairs_mask]]
        ], dim=1)
        return label_pairs, valid_pairs_mask
    

class FeatureDataset(Dataset):
    def __init__(self, features, labels, counts, batchsize):
        self.features = features
        self.labels = labels
        self.counts = counts
        self.batchsize = batchsize
        self.num_samples = len(self.features)

    def __len__(self):
        # Number of batches, not number of samples
        return (len(self.features) + self.batchsize - 1) // self.batchsize

    def __getitem__(self, idx):
        # Adjust idx to return a batch of data
        start_idx = idx * self.batchsize
        end_idx = min(start_idx + self.batchsize, self.num_samples)
        batch_features = self.features[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx]
        batch_counts = self.counts[start_idx:end_idx]
        indexes = torch.arange(start_idx, start_idx + batch_features.shape[0])
        return batch_features, batch_labels, batch_counts, indexes

def custom_collate_fn(batch):
    # Batch is a list of tuples: (features, labels, counts, idx)
    return batch[0]

def compute_similarity(features, labels, counts, batch_size, similarity_threshold, ambiguous_threshold):
    dataset = FeatureDataset(features, labels, counts, batch_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False, collate_fn=custom_collate_fn)
    indexes = torch.arange(len(features))
    model = SimilarityModel(similarity_threshold=similarity_threshold, ambiguous_threshold=ambiguous_threshold, 
                            features=features, labels=labels, counts=counts, indexes=indexes)
    num_gpus = torch.cuda.device_count()
    model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus)))
    model.eval()

    similar_pairs = []
    ambiguous_pairs = []

    for features1, labels1, counts1, indexes1 in tqdm(dataloader, desc='Computing similarity', ncols=75, miniters=1, total=len(dataloader)):
        similar_pairs_batch, ambiguous_pairs_batch = model(features1, labels1, counts1, indexes1)
        similar_pairs.extend(similar_pairs_batch.cpu().numpy().tolist())
        ambiguous_pairs.extend(ambiguous_pairs_batch.cpu().numpy().tolist())

    return similar_pairs, ambiguous_pairs