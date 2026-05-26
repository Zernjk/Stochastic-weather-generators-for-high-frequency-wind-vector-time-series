"""
`Dataset` (pytorch) class is defined.
"""
from typing import Union
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

from utils import get_root_dir


class DatasetImporter(object):
    def __init__(self, data_scaling:bool=True, **kwargs):
        # training and test datasets
        # typically, you'd load the data, for example, using pandas

        self.data_root = get_root_dir().joinpath("datasets", 'training_imputed_full_days.npz') #474days
        wd_dat = np.load(self.data_root)['data']

        self.X_train = wd_dat[:470]
        self.Y_train = np.random.randint(0, 1, size=(470,1))  # X:(n_training_samples, 1, ts_length); 1 denotes a univariate time series; 2 classes in this example
        self.X_test = wd_dat[470:]
        self.Y_test = np.random.randint(0, 1, size=(4,1))  # (n_test_samples, 1, ts_length)

        if data_scaling:
            self.mean = np.nanmean(self.X_train, axis=(0, 2))[None,:,None]  # (1 c 1)
            self.std = np.nanstd(self.X_train, axis=(0, 2))[None,:,None]  # (1 c 1)
            self.X_train = (self.X_train - self.mean) / self.std  # (b c l)
            self.X_test = (self.X_test - self.mean) / self.std  # (b c l)

        np.nan_to_num(self.X_train, copy=False)
        np.nan_to_num(self.X_test, copy=False)

class DatasetImporter_ws(object):
    def __init__(self, data_scaling:bool=False, **kwargs):
        # training and test datasets
        # typically, you'd load the data, for example, using pandas

        self.data_root = get_root_dir().joinpath("datasets", 'training_imputed_full_days_ws.npz') #474days
        wd_dat = np.load(self.data_root)['data']

        self.X_train = wd_dat[:470, :2]
        self.Y_train = np.random.randint(0, 1, size=(470,1))  # X:(n_training_samples, 1, ts_length); 1 denotes a univariate time series; 2 classes in this example
        self.ws_train = wd_dat[:470, 2]
        self.X_test = wd_dat[470:, :2]
        self.Y_test = np.random.randint(0, 1, size=(4,1))  # (n_test_samples, 1, ts_length)
        self.ws_test = wd_dat[470:, 2]

        if data_scaling:
            self.mean = np.nanmean(self.X_train, axis=(0, 2))[None,:,None]  # (1 c 1)
            self.std = np.nanstd(self.X_train, axis=(0, 2))[None,:,None]  # (1 c 1)
            self.X_train = (self.X_train - self.mean) / self.std  # (b c l)
            self.X_test = (self.X_test - self.mean) / self.std  # (b c l)

        np.nan_to_num(self.X_train, copy=False)
        np.nan_to_num(self.X_test, copy=False)


class DatasetImporter_consecutive(object):
    def __init__(self, data_scaling:bool=True, **kwargs):
        # training and test datasets
        # typically, you'd load the data, for example, using pandas

        self.data_root = get_root_dir().joinpath("datasets", 'training_imputed_full_2days.npz')
        # _2days.npz shape: (n_days=444, 2, 2800)
        wd_dat = np.load(self.data_root)['data']

        self.X_train = wd_dat[:440]
        self.Y_train = np.random.randint(0, 1, size=(440,1))  # X:(n_training_samples, 1, ts_length); 1 denotes a univariate time series; 2 classes in this example
        self.X_test = wd_dat[440:]
        self.Y_test = np.random.randint(0, 1, size=(4,1))  # (n_test_samples, 1, ts_length)

        if data_scaling:
            self.mean = np.nanmean(self.X_train, axis=(0, 2))[None,:,None]  # (1 c 1)
            self.std = np.nanstd(self.X_train, axis=(0, 2))[None,:,None]  # (1 c 1)
            self.X_train = (self.X_train - self.mean) / self.std  # (b c l)
            self.X_test = (self.X_test - self.mean) / self.std  # (b c l)

        np.nan_to_num(self.X_train, copy=False)
        np.nan_to_num(self.X_test, copy=False)     

class DatasetImporter_consecutive_ws(object):
    def __init__(self, data_scaling:bool=True, **kwargs):
        # training and test datasets
        # typically, you'd load the data, for example, using pandas

        self.data_root = get_root_dir().joinpath("datasets", 'training_imputed_full_2days_ws.npz')
        # _2days.npz shape: (n_days=444, 2, 2800)
        wd_dat = np.load(self.data_root)['data']

        self.X_train = wd_dat[:440, :2]
        self.Y_train = np.random.randint(0, 1, size=(440,1))  # X:(n_training_samples, 1, ts_length); 1 denotes a univariate time series; 2 classes in this example
        self.ws_train = wd_dat[:440, 2]
        self.X_test = wd_dat[440:, :2]
        self.Y_test = np.random.randint(0, 1, size=(4,1))  # (n_test_samples, 1, ts_length)
        self.ws_test = wd_dat[440:, 2]

        if data_scaling:
            self.mean = np.nanmean(self.X_train, axis=(0, 2))[None,:,None]  # (1 c 1)
            self.std = np.nanstd(self.X_train, axis=(0, 2))[None,:,None]  # (1 c 1)
            self.X_train = (self.X_train - self.mean) / self.std  # (b c l)
            self.X_test = (self.X_test - self.mean) / self.std  # (b c l)

        np.nan_to_num(self.X_train, copy=False)
        np.nan_to_num(self.X_test, copy=False)     
        


class WindDataset(Dataset):
    def __init__(self, kind: str, dataset_importer:DatasetImporter, **kwargs):
        """
        :param kind: "train" | "test"
        :param dataset_importer: instance of the `DatasetImporter` class.
        """
        super().__init__()
        kind = kind.lower()
        assert kind in ['train', 'test']
        self.kind = kind
        
        if kind == "train":
            self.X, self.Y = dataset_importer.X_train, dataset_importer.Y_train
        elif kind == "test":
            self.X, self.Y = dataset_importer.X_test, dataset_importer.Y_test
        else:
            raise ValueError

        self._len = self.X.shape[0]
    
    def __getitem__(self, idx):
        x, y = self.X[idx, :], self.Y[idx, :]
        return x, y

    def __len__(self):
        return self._len
    
class WindDataset_ws(Dataset):
    def __init__(self, kind: str, dataset_importer:DatasetImporter, **kwargs):
        """
        :param kind: "train" | "test"
        :param dataset_importer: instance of the `DatasetImporter` class.
        """
        super().__init__()
        kind = kind.lower()
        assert kind in ['train', 'test']
        self.kind = kind
        
        if kind == "train":
            self.X, self.Y, self.ws = dataset_importer.X_train, dataset_importer.Y_train, dataset_importer.ws_train
        elif kind == "test":
            self.X, self.Y, self.ws = dataset_importer.X_test, dataset_importer.Y_test, dataset_importer.ws_test
        else:
            raise ValueError

        self._len = self.X.shape[0]
    
    def __getitem__(self, idx):
        x, y, ws = self.X[idx, :], self.Y[idx, :], self.ws[idx, :]
        return x, y, ws

    def __len__(self):
        return self._len


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    import torch
    from torch.utils.data import DataLoader
    os.chdir("../")

    # data pipeline for custom data
    dataset_importer = DatasetImporter("ScreenType", data_scaling=True)
    dataset = WindDataset("train", dataset_importer)
    data_loader = DataLoader(dataset, batch_size=16, num_workers=0, shuffle=False)


    # get a mini-batch of samples
    for batch in data_loader:
        x, y = batch
        break
    print('x.shape:', x.shape)
    print('y.shape:', y.shape)
    print(y.flatten())