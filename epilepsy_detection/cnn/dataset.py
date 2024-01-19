import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torch
import zipfile

class WindowLevel(Dataset):

    def __init__(self, data_dir=None, transforms = None, split_level = 'window'):
        self.data_dir = data_dir
        self.transforms = transforms

        self.windows = []
        self.labels = []
        self.groups = []
        counter = 0
        if data_dir is not None:
            for file_name in os.listdir(data_dir):
                
                if counter == 5:
                    break
                
                if file_name.endswith(".npz"):
                    print(file_name)
                    npz_path = os.path.join(data_dir, file_name)
                    # print(npz_path)
                    # if "chb01_seizure_EEGwindow_1" not in npz_path:
                    #     print("Skipping file: ", npz_path)
                    #     continue
                    parquet_file_name = file_name.replace("seizure_EEGwindow", "seizure_metadata").replace(".npz", ".parquet")
                    parquet_path = os.path.join(data_dir, parquet_file_name)

                    # If the Parquet file is inside a zip archive
                    if not os.path.exists(parquet_path) and "MetaData.zip" in os.listdir(data_dir):
                        zip_path = os.path.join(data_dir, "MetaData.zip")
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extract(parquet_file_name, data_dir)
                    
                    try:
                        npz_data = np.load(npz_path, allow_pickle=True)['EEG_win']
                    except:
                        print("Error loading file: ", npz_path)
                        continue
                    try:
                        parquet_data = pd.read_parquet(parquet_path)
                    except:
                        print("Error loading file: ", parquet_path)
                        continue
                    # print(npz_path, parquet_path)
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
        # print(f"window.shape: {window.shape}, label.shape {label}")
        label_ohe = torch.eye(2)[label].float()

        return window, label_ohe

if __name__ == "__main__":
    a = WindowLevel('../data/annotated_windows/')
    print(len(a))