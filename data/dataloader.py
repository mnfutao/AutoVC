from torch.utils.data.dataloader import default_collate, DataLoader
import numpy as np


class VCDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, mode='train', num_workers=0):
        super().__init__(dataset,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=collate_fn,
                         shuffle=True)

def collate_fn(batch):
    
    keys = batch[0].keys()
    max_lengths = {key: 0 for key in keys}
    collated_batch = {key: [] for key in keys}

    # find out the max lengths
    for row in batch:
        for key in keys:
            max_lengths[key] = max(max_lengths[key], row[key].shape[0])

    # pad to the max lengths
    for row in batch:
        for key in keys:
            array = row[key]
            dim = len(array.shape)
            assert dim == 1 or dim == 2
            # TODO: because of pre processing, later we want to have (n_mels, T)
            if dim == 1:
                padded_array = np.pad(array, (0, max_lengths[key] - array.shape[0]), mode='constant')
            else:
                padded_array = np.pad(array, ((0, max_lengths[key] - array.shape[0]), (0, 0)), mode='constant')
            collated_batch[key].append(padded_array)

    # use the default_collate to convert to tensors
    for key in keys:
        collated_batch[key] = default_collate(collated_batch[key])
    return collated_batch
