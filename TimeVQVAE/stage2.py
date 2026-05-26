"""
Stage2: prior learning

run `python stage2.py`
"""
import os
import copy
from argparse import ArgumentParser
import argparse

import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader
from preprocessing.data_pipeline import *
from preprocessing.preprocess import *
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from experiments.exp_stage2 import ExpStage2
from utils import get_root_dir, load_yaml_param_settings, str2bool


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.",
                        default=get_root_dir().joinpath('configs', 'config.yaml'))
    parser.add_argument('--dataset_name_stage1', default='Wind')
    parser.add_argument('--dataset_name', default='Wind')
    parser.add_argument('--gpu_device_ind', nargs='+', default=[0], type=int, help='Indices of GPU devices to use.')
    parser.add_argument('--use_weather_states', type=str2bool, default=False, help='Using weather states information, then set it to True')
    parser.add_argument('--consecutive', type=str2bool, default=False, help='Set to True if train consecutive sampling')
    parser.add_argument('--weather_states_mask', type=str2bool, default=False, help='Set to True if want to mask the weather states during trainning')
    
    return parser.parse_args()


def train_stage2(config: dict,
                 dataset_name_stage1:str,
                 dataset_name: str,
                 train_data_loader: DataLoader,
                 test_data_loader: DataLoader,
                 gpu_device_ind,
                 use_weather_states:bool,
                 consecutive:bool,
                 weather_states_mask:bool,
                 ):
    project_name = 'TimeVQVAE-stage2'

    # fit
    n_classes = len(np.unique(train_data_loader.dataset.Y))
    _, in_channels, input_length = train_data_loader.dataset.X.shape
    train_exp = ExpStage2(dataset_name_stage1, in_channels, input_length, config, n_classes, use_weather_states, consecutive, weather_states_mask)
    
    n_trainable_params = sum(p.numel() for p in train_exp.parameters() if p.requires_grad)
    wandb_logger = WandbLogger(project=project_name, name=None, 
                               config={**config, 'dataset_name': dataset_name, 'n_trainable_params': n_trainable_params})
    
    # Check if GPU is available
    if not torch.cuda.is_available():
        print('GPU is not available.')
        # num_cpus = multiprocessing.cpu_count()
        num_cpus = 1
        print(f'using {num_cpus} CPUs..')
        device = num_cpus
        accelerator = 'cpu'
    else:
        accelerator = 'gpu'
        device = gpu_device_ind

    trainer = pl.Trainer(logger=wandb_logger,
                         enable_checkpointing=False,
                         callbacks=[LearningRateMonitor(logging_interval='step')],
                         max_steps=config['trainer_params']['max_steps']['stage2'],
                         devices=device,
                         accelerator=accelerator,
                         val_check_interval=config['trainer_params']['val_check_interval']['stage2'],
                         check_val_every_n_epoch=None,
                         # precision='bf16',
                         accumulate_grad_batches=1,
                         )
    trainer.fit(train_exp,
                train_dataloaders=train_data_loader,
                val_dataloaders=test_data_loader
                )

    print('saving the model...')
    if not os.path.isdir(get_root_dir().joinpath('saved_models')):
        os.mkdir(get_root_dir().joinpath('saved_models'))
    trainer.save_checkpoint(os.path.join(f'saved_models', f'stage2-{dataset_name}.ckpt'))

    wandb.finish()


if __name__ == '__main__':
    # load config
    args = load_args()
    config = load_yaml_param_settings(args.config)

    # config
    dataset_name = args.dataset_name
    dataset_name_stage1 = args.dataset_name_stage1

    # run
    
    # data pipeline
    batch_size = config['dataset']['batch_sizes']['stage2']

    if args.use_weather_states:
        if args.consecutive:
            dataset_importer = DatasetImporter_consecutive_ws(**config['dataset'])
            train_data_loader, test_data_loader = [build_ws_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]
        else:
            dataset_importer = DatasetImporter_ws(**config['dataset'])
            train_data_loader, test_data_loader = [build_ws_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]
    elif args.consecutive:
        dataset_importer = DatasetImporter_consecutive(**config['dataset'])
        train_data_loader, test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]
    else:
        dataset_importer = DatasetImporter(**config['dataset'])
        train_data_loader, test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]

    # train
    train_stage2(config, dataset_name_stage1, dataset_name, train_data_loader, test_data_loader, args.gpu_device_ind, args.use_weather_states, args.consecutive, args.weather_states_mask)
