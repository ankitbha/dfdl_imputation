import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

class GRNDataset(torch.utils.data.Dataset):
    def __init__(self, data, adjacency_matrix):
        self.data = torch.FloatTensor(data)  # Shape: (400, 2700)
        self.adjacency_matrix = torch.FloatTensor(adjacency_matrix)  # Shape: (400, 400)

    def __len__(self):
        return self.data.shape[0]  # Number of genes

    def __getitem__(self, idx):
        return self.data[idx], self.adjacency_matrix[idx]

class GRNVAEDataset(LightningDataModule):
    def __init__(
        self,
        data: np.ndarray,
        adjacency_matrix: np.ndarray,
        train_val_test_split: tuple = (0.7, 0.15, 0.15),
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False
    ):
        super().__init__()
        self.data = data
        self.adjacency_matrix = adjacency_matrix
        self.train_val_test_split = train_val_test_split
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: str = None):
        # Create train, validation, and test splits
        num_samples = len(self.data)
        indices = np.random.permutation(num_samples)
        train_split, val_split, test_split = self.train_val_test_split
        train_end = int(train_split * num_samples)
        val_end = train_end + int(val_split * num_samples)

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        self.train_dataset = GRNDataset(self.data[train_indices], self.adjacency_matrix[train_indices])
        self.val_dataset = GRNDataset(self.data[val_indices], self.adjacency_matrix[val_indices])
        self.test_dataset = GRNDataset(self.data[test_indices], self.adjacency_matrix[test_indices])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )