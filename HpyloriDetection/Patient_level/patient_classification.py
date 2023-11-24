# Import libraries
import torch
import pandas as pd
from dataset import DatasetPatients
from torchvision import transforms
from ae import AEanomaly
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset

# Define dataset
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
csv_path = 'Data/short.csv'
data_dir = 'Data'
ds = DatasetPatients(data_dir, csv_path, transforms = data_transform)

# Load the autoencoder model
state_dict = torch.load('Autoencoder/model25E.pth', map_location = torch.device('cpu'))['model_state_dict']
model = AEanomaly()
model.load_state_dict(state_dict)
model.eval()

def red_pixels(img_array):
    pil_image = Image.fromarray(np.uint8(img_array.transpose(1,2,0)*255)).convert('RGB')
    hsv_img = pil_image.convert('HSV')
    pixels = list(hsv_img.getdata())
    red_like_range = (-20, 20)
    red_like_pixel_count = sum(1 for pixel in pixels if red_like_range[0] <= pixel[0] <= red_like_range[1])
    return red_like_pixel_count

# Loop through the data and classify patient
# Classifiquem pacients -> positiu (> 5 % de crops with h.pylori) i negatiu (< 5 % crop with h.pylori)
num_folds = 5
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

for train_index, val_index in skf.split(range(len(ds)), ds.labels):
    train_ds = Subset(ds, train_index)
    val_ds = Subset(ds, val_index)

    patient_label = []
    patient_label_prediction = []
    percentatge_positius = {'POSITIVE': [], 'NEGATIVE': []}
    label_original = {'NEGATIVA': [], 'ALTA': [], 'BAIXA': []}

    results_fold = []
    threshold = 298.6
    batch_size = 64
    for imgs, p_label in tqdm(train_ds, desc="Predicting for patient"):
        results = []
        threshold = 298.6
        batch_size = 64
        for i in tqdm(range(0, len(imgs), batch_size), desc="Predicting for patient"):
            image = torch.stack(imgs[i:i+batch_size])
            single_reconstructed = model(image)
            for j in range(single_reconstructed.shape[0]):
                red_pixels_reconstruction = red_pixels(single_reconstructed[j, :, :, :].detach().cpu().numpy())
                red_pixels_original = red_pixels(image[j,:,:,:].numpy())
                if red_pixels_original - red_pixels_reconstruction >= threshold:
                    results.append(1)
                else:
                    results.append(-1)
        
        results = np.array(results)
        num_pos = len(results[results == 1])
        patient_label.append('POSITIVE' if p_label in ['ALTA', 'BAIXA'] else 'NEGATIVE')
        if num_pos/len(results) > 0.05:
            patient_label_prediction.append('POSITIVE')
        else:
            patient_label_prediction.append('NEGATIVE')
        percentatge_positius[patient_label_prediction[-1]].append(num_pos/len(results))
        label_original[p_label].append(num_pos/len(results))

        for imgs, p_label in tqdm(val_ds, desc="Predicting for patient"):
            results = []
            for i in range(0, len(imgs), batch_size):
                image = torch.stack(imgs[i:i+batch_size])
                single_reconstructed = model(image)
                for j in range(single_reconstructed.shape[0]):
                    red_pixels_reconstruction = red_pixels(single_reconstructed[j, :, :, :].detach().cpu().numpy())
                    red_pixels_original = red_pixels(image[j,:,:,:].numpy())
                    if red_pixels_original - red_pixels_reconstruction >= threshold:
                        results.append(1)
                    else:
                        results.append(-1)

print("DONE")
