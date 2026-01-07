import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    """
    Custom dataset supporting both CD (Channel Dependency) and CI (Channel Independency) modes.

    Args:
        root_path: Root directory of the data
        flag: 'train', 'val', or 'test'
        size: [seq_len, label_len, pred_len]
        data_path: CSV file name
        features: List of feature names to use. None means use all features.
        target_features: For CI mode, which features to predict. None means all features.
        mode: 'CD' (all features in one model) or 'CI' (one feature per model)
        target_feature: For CI mode, which single feature to use for this dataset instance
        scale: Whether to apply standardization
        timeenc: 0 for manual time features, 1 for time_features function
        freq: Frequency string for time features
    """
    def __init__(self, root_path, flag='train', size=None,
                 data_path='data.csv', features=None, target_features=None,
                 mode='CD', target_feature=None, scale=True, timeenc=0, freq='h'):

        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.mode = mode
        self.features = features
        self.target_features = target_features
        self.target_feature = target_feature
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Assume first column is date
        cols = list(df_raw.columns)
        if 'date' in cols[0].lower():
            df_raw.rename(columns={cols[0]: 'date'}, inplace=True)
            cols_data = df_raw.columns[1:]
        else:
            cols_data = df_raw.columns[:]

        # Select features
        if self.features is not None:
            # Remove duplicates while preserving order
            seen = set()
            available_features = []
            for f in self.features:
                if f in cols_data and f not in seen:
                    available_features.append(f)
                    seen.add(f)
            cols_data = available_features
        else:
            cols_data = list(cols_data)

        # Store all feature names for reference
        self.all_features = list(cols_data)

        # For CI mode, use only the target feature
        if self.mode == 'CI':
            if self.target_feature is None:
                raise ValueError("target_feature must be specified for CI mode")
            cols_data = [self.target_feature]

        # Data split: 70% train, 10% val, 20% test
        num_train = int(len(df_raw) * 0.7)
        num_val = int(len(df_raw) * 0.1)
        num_test = len(df_raw) - num_train - num_val

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Ensure cols_data is a list
        cols_data = list(cols_data)
        df_data = df_raw[cols_data]

        # Scaling with feature-wise scalers
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scalers = {}

            # Get number of rows and columns explicitly
            n_rows = df_data.shape[0]
            n_cols = df_data.shape[1]

            # Create scaled_data with explicit shape
            scaled_data = np.zeros((n_rows, n_cols), dtype=np.float32)

            for i, col in enumerate(cols_data):
                scaler = StandardScaler()
                scaler.fit(train_data[col].values.reshape(-1, 1))
                transformed = scaler.transform(df_data[col].values.reshape(-1, 1)).ravel()
                scaled_data[:, i] = transformed
                self.scalers[col] = scaler

            data = scaled_data
        else:
            data = df_data.values.astype(np.float32)
            self.scalers = None

        # Time features
        if 'date' in df_raw.columns:
            df_stamp = df_raw[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)

            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(['date'], axis=1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)
        else:
            # No date column, use dummy time features
            data_stamp = np.zeros((border2 - border1, 4))

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data, feature_name=None):
        """
        Inverse transform the scaled data.

        Args:
            data: Scaled data to inverse transform
            feature_name: Name of the feature to inverse transform (for CI mode)

        Returns:
            Original scale data
        """
        if self.scalers is None:
            return data

        if self.mode == 'CI' and feature_name is None:
            feature_name = self.target_feature

        if feature_name is not None:
            return self.scalers[feature_name].inverse_transform(data)

        # For CD mode, inverse transform all features
        result = np.zeros_like(data)
        for i, col in enumerate(self.all_features if self.mode == 'CD' else [self.target_feature]):
            result[..., i] = self.scalers[col].inverse_transform(data[..., i:i+1]).ravel()
        return result
