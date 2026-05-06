import torch
import numpy as np
import json

from torch.utils.data import Dataset


class NumpyDataset(Dataset):
    def __init__(self, X_path, y_path, features_path,filter_noise=True,min_move = 0.004):
        self.X = torch.FloatTensor(np.load(X_path))
        self.y_r8 = torch.FloatTensor(np.load(y_path))
        MODEL_EXCLUDE = {'Close', 'High', 'Low'}


        # Generate binary labels (1 if future price > current price)
        # X shape: (N, T, F)
        # current_price is the Close price at the last time step of the window

        price_diff = self.y_r8
        if filter_noise:
            valid_mask = price_diff.abs() > min_move

            self.X = self.X[valid_mask]
            self.y_r8 = self.y_r8[valid_mask]
            price_diff = price_diff[valid_mask]
        self.y_class = (price_diff > 0).float()
        # Calculate price change ratio for regression target
        # Avoid division by zero
        self.y_change =  price_diff



    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_class[idx].unsqueeze(0), self.y_change[idx].unsqueeze(0)