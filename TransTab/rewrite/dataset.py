import torch

from torch.utils.data import Dataset, DataLoader

class TableDataset(Dataset):
    def __init__(self, dataset):
        self.x, self.y = dataset
        self.length = len(self.x)

    def __len__(self):
        return self.length

    def __getitems__(self, indices):
        x = self.x.iloc[indices]
        y = self.y.iloc[indices]
        y = torch.tensor(y.values, dtype=torch.long)
        return x, y

def get_loader(dataset, batch_size, shuffle=True):
    dataset = TableDataset(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=lambda x: x,
        pin_memory=True
    )
    return loader
