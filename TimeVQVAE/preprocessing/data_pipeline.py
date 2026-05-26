from torch.utils.data import DataLoader
from preprocessing.preprocess import *

def build_data_pipeline(batch_size, dataset_importer:DatasetImporter, config: dict, kind: str) -> DataLoader:
    """
    :param config:
    :param kind train/valid/test
    """
    num_workers = config['dataset']["num_workers"]

    # DataLoader
    if kind == 'train':
        wind_dataset = WindDataset('train', dataset_importer)
        return DataLoader(wind_dataset, batch_size, num_workers=num_workers, shuffle=True, drop_last=False, pin_memory=True)  # `drop_last=False` due to some datasets with a very small dataset size.
    elif kind == 'test':
        wind_dataset = WindDataset('test', dataset_importer)
        return DataLoader(wind_dataset, batch_size, num_workers=num_workers, shuffle=False, drop_last=False, pin_memory=True)  # `drop_last=False` due to some datasets with a very small dataset size.
    else:
        raise ValueError


def build_ws_data_pipeline(batch_size, dataset_importer:DatasetImporter_ws, config: dict, kind: str) -> DataLoader:
    """
    :param config:
    :param kind train/valid/test
    """
    num_workers = config['dataset']["num_workers"]

    # DataLoader
    if kind == 'train':
        custom_dataset = WindDataset_ws('train', dataset_importer)
        return DataLoader(custom_dataset, batch_size, num_workers=num_workers, shuffle=True, drop_last=False, pin_memory=True)  # `drop_last=False` due to some datasets with a very small dataset size.
    elif kind == 'test':
        custom_dataset = WindDataset_ws('test', dataset_importer)
        return DataLoader(custom_dataset, batch_size, num_workers=num_workers, shuffle=False, drop_last=False, pin_memory=True)  # `drop_last=False` due to some datasets with a very small dataset size.
    else:
        raise ValueError
