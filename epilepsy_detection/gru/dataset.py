import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torch
import zipfile

class WindowLevel(Dataset):

    def __init__(self, data_dir, transforms = None, split_level = 'window'):
        self.data_dir = data_dir
        self.transforms = transforms

        self.windows = []
        self.labels = []
        self.groups = []
        counter = 0

        for file_name in os.listdir(data_dir):
            if counter == 5:
                break
            if file_name.endswith(".npz"):
                npz_path = os.path.join(data_dir, file_name)
            
                parquet_file_name = file_name.replace("seizure_EEGwindow", "seizure_metadata").replace(".npz", ".parquet")
                parquet_path = os.path.join(data_dir, parquet_file_name)

                # If the Parquet file is inside a zip archive
                if not os.path.exists(parquet_path) and "MetaData.zip" in os.listdir(data_dir):
                    zip_path = os.path.join(data_dir, "MetaData.zip")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extract(parquet_file_name, data_dir)

                npz_data = np.load(npz_path, allow_pickle=True)['EEG_win']

                parquet_data = pd.read_parquet(parquet_path)
                if split_level == 'seizure':
                    parquet_data['grup'] = parquet_data['filename']+parquet_data['global_interval'].astype(str)
                elif split_level == 'patient':
                    parquet_data['grup'] = parquet_data['filename'].apply(lambda x: x.split('_')[0])
                else:
                    parquet_data['grup'] = 0
                labels = parquet_data['class'].values
                grups = parquet_data['grup'].values

                self.windows.extend(npz_data)
                self.labels.extend(labels)
                self.groups.extend(grups)
                counter += 1
        
        self.windows = np.array(self.windows)
        self.labels = np.array(self.labels)
    
    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        window = self.windows[index]
        label = self.labels[index]

        window = window.astype(np.float32)

        if self.transforms:
            window = self.transforms(window)

        window = torch.from_numpy(window)
        label = torch.tensor(label)
        label_ohe = torch.eye(2)[label].float()

        return window, label_ohe