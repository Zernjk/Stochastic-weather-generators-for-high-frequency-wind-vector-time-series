from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random
from scipy import signal
from const import EVALUATION_DATA_PATH, NO_WEATHER_CONDITIONAL, WEATHER_FEATURE_CONDITIONAL, WEATHER_EMBED_MASK_CONDITIONAL, UNCONDITIONAL_TRAINING_DATA_PATH, VALIDATION_DATA_PATH

TRAINING_DATA_LENGTH = 474


def load_data(data_name):
    match data_name:
        case "training":
            data = np.load(UNCONDITIONAL_TRAINING_DATA_PATH)  # (474, 2, 1440)
            return np.transpose(data["data"], (0, 2, 1))  # (474, 1440, 2)
        case "validation":
            val_data = pd.read_csv(VALIDATION_DATA_PATH)
            val_data = np.array(val_data[["Easterly", "Northerly"]])
            val_data = val_data.reshape(-1, 1440, 2)
            return val_data  # (205, 1440, 2)
        case "testing":
            data = pd.read_csv(EVALUATION_DATA_PATH)
            data = np.array(data[["Easterly", "Northerly"]])
            data = data.reshape(-1, 1440, 2)
            return data  # (185, 1440, 2)
        
        case "add_weather_consecutive":  # ! WEATHER STATE AS FEATURES
            data = np.load(WEATHER_FEATURE_CONDITIONAL) # (230, 3, 30240)
            data = data["data"][:, :2, :].reshape(230, 2, 21, 1440)
            data = np.transpose(data, (0, 2, 3, 1))  # (230, 21, 1440, 2)
            return data.reshape(230 * 21, 1440, 2)  # (4830, 1440, 2)
        
        case "add_weather_emb_mask_consecutive":  # ! WEATHER STATE AS EMBEDDINGS 
            data = np.load(WEATHER_EMBED_MASK_CONDITIONAL)  # (230, 2, 30240)
            data = data["data"].reshape(230, 2, 21, 1440)
            data = np.transpose(data, (0, 2, 3, 1))  # (230, 21, 1440, 2)
            return data.reshape(230 * 21, 1440, 2)  # (4830, 1440, 2)

        case "generated_consecutive":
            data = np.load(NO_WEATHER_CONDITIONAL)  # (230, 2, 30240)
            data = data["data"].reshape(230, 2, 21, 1440)
            data = np.transpose(data, (0, 2, 3, 1))  # (230, 21, 1440, 2)
            return data.reshape(230 * 21, 1440, 2)  # (4830, 1440, 2)
        
        case _:
            raise ValueError(f"Invalid data name: {data_name}")
        
        
def get_unwanted_indices(gen_data):
    match gen_data.shape:
        case (474, 1440, 2):
            unwanted_indices = [
                70, 71, 72, 73, 157, 158, 288, 289, 440
            ]  # ! training data missing
            
        case (205, 1440, 2):
            unwanted_indices = [
                11, 152
            ]  # ! validation data missing
            
        case (185, 1440, 2):
            unwanted_indices = [
                16, 22, 67, 83
            ]  # ! testing data missing
            
        case (4830, 1440, 2):
            unwanted_indices = [
                637, 638, 639, 640, 658, 659, 660, 661, 679, 680, 681, 682, 700, 701, 702, 703, 721, 722, 723, 724, 742, 743, 744, 745, 763, 764, 765, 766, 784, 785, 786, 787, 805, 806, 807, 808, 826, 827, 828, 829, 1480, 1481, 1501, 1502, 1522, 1523, 1543, 1544, 1564, 1565, 1585, 1586, 1606, 1607, 1627, 1628, 1648, 1649, 1669, 1670, 2745, 2746, 2766, 2767, 2787, 2788, 2808, 2809, 2829, 2830, 2850, 2851, 2871, 2872, 2892, 2893, 2913, 2914, 2934, 2935, 4220, 4241, 4262, 4283, 4304, 4325, 4346, 4367, 4388, 4409
            ]
            
        case _:
            raise ValueError(f"Invalid data shape: {gen_data.shape}")
    return unwanted_indices


def get_three_day_wind_data(data):
    unwanted_indices = get_unwanted_indices(data)
    three_day_sequences = []

    match data.shape:
        case (474, 1440, 2) | (185, 1440, 2):  # training or testing data
            # Reconstruct original indices (data has unwanted indices already removed)
            original_length = len(data) + len(unwanted_indices)
            assert original_length % 21 == 0, "Original length must be divisible by 21"
            num_periods = original_length // 21

            # Create mapping: original_index -> current_index
            current_idx = 0
            original_to_current = {}
            for orig_idx in range(original_length):
                if orig_idx not in unwanted_indices:
                    original_to_current[orig_idx] = current_idx
                    current_idx += 1

            # Extract 3-day sequences from each 21-day period
            for period in range(num_periods):
                start_idx = period * 21
                for i in range(19):  # 19 possible 3-day sequences in 21 days (0-2, 1-3, ..., 18-20)
                    day1_orig = start_idx + i
                    day2_orig = start_idx + i + 1
                    day3_orig = start_idx + i + 2

                    # Skip if any day is in unwanted_indices
                    if (day1_orig not in unwanted_indices and
                        day2_orig not in unwanted_indices and
                        day3_orig not in unwanted_indices):
                        # Map to current indices
                        day1_curr = original_to_current[day1_orig]
                        day2_curr = original_to_current[day2_orig]
                        day3_curr = original_to_current[day3_orig]

                        # Concatenate three consecutive days along time axis
                        three_day_seq = np.concatenate([
                            data[day1_curr],
                            data[day2_curr],
                            data[day3_curr]
                        ], axis=0)  # Shape: (4320, 2)

                        three_day_sequences.append(three_day_seq)

        case (205, 1440, 2):  # validation data
            # Reconstruct original indices
            original_length = len(data) + len(unwanted_indices)
            assert original_length % 9 == 0, "Original length must be divisible by 9"
            num_periods = original_length // 9

            # Create mapping: original_index -> current_index
            current_idx = 0
            original_to_current = {}
            for orig_idx in range(original_length):
                if orig_idx not in unwanted_indices:
                    original_to_current[orig_idx] = current_idx
                    current_idx += 1

            # Extract 3-day sequences from each 21-day period
            for period in range(num_periods):
                start_idx = period * 9
                for i in range(7):
                    day1_orig = start_idx + i
                    day2_orig = start_idx + i + 1
                    day3_orig = start_idx + i + 2

                    if (day1_orig not in unwanted_indices and
                        day2_orig not in unwanted_indices and
                        day3_orig not in unwanted_indices):
                        day1_curr = original_to_current[day1_orig]
                        day2_curr = original_to_current[day2_orig]
                        day3_curr = original_to_current[day3_orig]

                        three_day_seq = np.concatenate([
                            data[day1_curr],
                            data[day2_curr],
                            data[day3_curr]
                        ], axis=0)

                        three_day_sequences.append(three_day_seq)

        case (4830, 1440, 2):  # generation data (already in 21*N shape)
            # Data is already structured in 21-day periods
            assert len(data) % 21 == 0, "Data length must be divisible by 21"
            num_periods = len(data) // 21

            for period in range(num_periods):
                start_idx = period * 21
                for i in range(19):  # 19 possible 3-day sequences
                    day1_idx = start_idx + i
                    day2_idx = start_idx + i + 1
                    day3_idx = start_idx + i + 2

                    # Skip if any day is in unwanted_indices
                    if (day1_idx not in unwanted_indices and
                        day2_idx not in unwanted_indices and
                        day3_idx not in unwanted_indices):
                        three_day_seq = np.concatenate([
                            data[day1_idx],
                            data[day2_idx],
                            data[day3_idx]
                        ], axis=0)  # Shape: (4320, 2)

                        three_day_sequences.append(three_day_seq)

        case _:
            raise ValueError(f"Invalid data shape: {data.shape}")

    return np.array(three_day_sequences)
    
            
        
class DatasetGenerator():
    class ThreeDayWindDataset(Dataset):
        def __init__(self, data, dataset_type):
            self.data = data
            print(f"There are {len(self.data)} 3-day sequences in the {dataset_type} data")

        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, index):
            return np.array(self.data[index][0], dtype=np.float32), int(self.data[index][1])

    def __init__(self, real_name, gen_name, random_seed):
        super().__init__()

        if real_name == gen_name:
            raise ValueError("Real and generated data cannot be the same")

        self.real_name = real_name
        self.gen_name = gen_name
        self.random_seed = random_seed

        self.real_data = load_data(real_name)
        self.gen_data = load_data(gen_name)
        self.validation_data = load_data("validation")
        self.testing_data = load_data("testing")
        
        self.real_one_day_data = get_three_day_wind_data(self.real_data)
        print(f"There are {len(self.real_one_day_data)} 1-day sequences in the real data with shape {self.real_one_day_data.shape}")
        self.gen_one_day_data = get_three_day_wind_data(self.gen_data)
        print(f"There are {len(self.gen_one_day_data)} 1-day sequences in the generated data with shape {self.gen_one_day_data.shape}")
        self.validation_one_day_data = get_three_day_wind_data(self.validation_data)
        print(f"There are {len(self.validation_one_day_data)} 1-day sequences in the validation data with shape {self.validation_one_day_data.shape}")
        self.testing_one_day_data = get_three_day_wind_data(self.testing_data)
        print(f"There are {len(self.testing_one_day_data)} 1-day sequences in the testing data with shape {self.testing_one_day_data.shape}")

        # Randomly sample self.length data points from each dataset
        np.random.seed(self.random_seed)  # Set seed for reproducibility
        self.gen_for_training_indices = np.random.choice(len(self.gen_one_day_data), len(self.real_one_day_data), replace=False)

        gen_for_validation_indices_temp = list(set(range(0, len(self.gen_one_day_data))) - set(self.gen_for_training_indices))
        self.gen_for_validation_indices = np.random.choice(gen_for_validation_indices_temp, len(self.validation_one_day_data), replace=False)

        gen_for_testing_indices_temp = list(set(range(0, len(self.gen_one_day_data))) - set(self.gen_for_training_indices) - set(self.gen_for_validation_indices))
        self.gen_for_testing_indices = np.random.choice(gen_for_testing_indices_temp, len(self.testing_one_day_data), replace=False)

        self.training_data = [(d, 1) for d in self.real_one_day_data] + [(self.gen_one_day_data[i], 0) for i in self.gen_for_training_indices]
        self.validation_data = [(d, 1) for d in self.validation_one_day_data] + [(self.gen_one_day_data[i], 0) for i in self.gen_for_validation_indices]
        self.testing_data = [(d, 1) for d in self.testing_one_day_data] + [(self.gen_one_day_data[i], 0) for i in self.gen_for_testing_indices]
        
        print(f"There are {len(self.training_data)} 1-day sequences in the training data")
        print(f"There are {len(self.validation_data)} 1-day sequences in the validation data")
        print(f"There are {len(self.testing_data)} 1-day sequences in the testing data")
        
        random.seed(self.random_seed)
        random.shuffle(self.training_data)

    def get_dataset(self, train_or_test):
        if train_or_test == "training":
            return self.ThreeDayWindDataset(self.training_data, dataset_type="training")
        elif train_or_test == "validation":
            return self.ThreeDayWindDataset(self.validation_data, dataset_type="validation")
        elif train_or_test == "testing":
            return self.ThreeDayWindDataset(self.testing_data, dataset_type="testing")
        else:
            raise ValueError(f"Invalid dataset type: {train_or_test}")
