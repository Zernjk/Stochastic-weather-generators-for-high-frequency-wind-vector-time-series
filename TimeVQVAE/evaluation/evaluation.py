import os
from typing import List, Union, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn
import wandb
import numpy as np
import matplotlib.pyplot as plt

from experiments.exp_stage2 import ExpStage2
from generators.maskgit import MaskGIT
from generators.maskgit_consecutive import MaskGIT_consecutive
from preprocessing.data_pipeline import build_data_pipeline
from preprocessing.preprocess import DatasetImporter
from generators.sample import *



class Evaluation(nn.Module):
    """
    do sampling after training
    """
    def __init__(self, 
                 dataset_name_stage1:str,
                 dataset_name: str, 
                 in_channels:int,
                 input_length:int, 
                 n_classes:int, 
                 use_weather_states:bool,
                 consecutive:bool,
                 weather_states_mask:bool,
                 device:int, 
                 config:dict
                 ):
        super().__init__()
        self.dataset_name_stage1 = dataset_name_stage1
        self.dataset_name = dataset_name
        self.device = torch.device(device)
        self.config = config
        self.batch_size = self.config['evaluation']['batch_size']

        # load the numpy matrix of the test samples
        dataset_importer = DatasetImporter(**config['dataset'])
        self.X_train = dataset_importer.X_train
        self.X_test = dataset_importer.X_test
        self.Y_train = dataset_importer.Y_train
        self.Y_test = dataset_importer.Y_test
        
        if config['dataset']['data_scaling']:
            self.mean = dataset_importer.mean  # scaling coefficient
            self.std = dataset_importer.std  # scaling coefficient
        else:
            self.mean = None
            self.std = None

        self.ts_len = self.X_train.shape[-1]  # time series length
        self.n_classes = len(np.unique(dataset_importer.Y_train))

        # load the stage2 model
        self.stage2 = ExpStage2.load_from_checkpoint(os.path.join('saved_models', f'stage2-{dataset_name}.ckpt'), 
                                                      dataset_name=dataset_name_stage1, 
                                                      in_channels=in_channels,
                                                      input_length=input_length, 
                                                      config=config,
                                                      n_classes=n_classes,
                                                      use_weather_states=use_weather_states,
                                                      consecutive = consecutive,
                                                      weather_states_mask=weather_states_mask,
                                                      map_location='cpu',
                                                      strict=True)
        self.stage2.eval()
        self.maskgit = self.stage2.maskgit
        self.stage1 = self.stage2.maskgit.stage1

    @torch.no_grad()
    def sample(self, n_samples: int, kind: str, class_index:Union[int,None]=None, unscale:bool=False, batch_size=None, 
               return_representations=False, s_before=None, ws_before=None, weather_states=None, weather_states_mask=False):
       
        assert kind in ['unconditional', 'conditional', 'consecutive']

        # sampling
        if kind == 'unconditional':
            res = unconditional_sample(self.maskgit, n_samples, self.device, batch_size=batch_size if not isinstance(batch_size, type(None)) else self.batch_size, weather_states=weather_states, weather_states_mask=weather_states_mask)  # (b c l); b=n_samples, c=1 (univariate)
        elif kind == 'conditional':
            res = conditional_sample(self.maskgit, n_samples, self.device, class_index, batch_size=batch_size if not isinstance(batch_size, type(None)) else self.batch_size, weather_states=weather_states, weather_states_mask=weather_states_mask)  # (b c l); b=n_samples, c=1 (univariate)
        elif kind == 'consecutive':
            res = consecutive_sample(self.maskgit, n_samples, self.device, class_index,
                                     batch_size=batch_size if not isinstance(batch_size, type(None)) else self.batch_size,
                                     return_representations=return_representations, 
                                     k=self.config['MaskGIT']['connect_period'], 
                                     s_before=s_before,
                                     ws_before=ws_before,
                                     weather_states=weather_states,
                                     weather_states_mask=weather_states_mask)
            # res = x_new, quantize_new, s_new, ws_new
        else:
            raise ValueError

        # unscale
        if unscale:
            res[0] = res[0]*self.std + self.mean
        
        return res    