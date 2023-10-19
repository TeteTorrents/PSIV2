from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


video_path = r'car_tracker\videos\mini2.mp4'
cap = cv2.VideoCapture(video_path)

frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

cap.release()

train_frames, test_frames = train_test_split(frames, train_size=0.6, shuffle=False)
train_frames = np.stack(train_frames)
test_frames = np.stack(test_frames)

print(f'Frames de train: {train_frames.shape}')
print(f'Frames de test: {test_frames.shape}')

gmm_background = np.zeros(shape = (train_frames.shape[1], train_frames.shape[2], 3))
for i in tqdm(range(train_frames.shape[1])):
    for j in range(train_frames.shape[2]):
        X = train_frames[:, i, j, :]
        X = X.reshape(X.shape[0], 3)
        gmm = GaussianMixture(2,max_iter=5)
        gmm.fit(X)
        means = gmm.means_
        weights = gmm.weights_
        idx = np.argmax(weights)
        gmm_background[i][j][:] = means[idx]


max_value = gmm_background.max()
gmm_background[gmm_background == max_value] = 255
gmm_background /= 255
plt.imshow(gmm_background)
plt.show()

for i in range(test_frames.shape[0]):
    foregrounds = np.abs(test_frames[i]*255 - gmm_background*255)
    foregrounds = np.array(foregrounds,dtype='uint8')
    foregrounds = cv2.cvtColor(foregrounds, cv2.COLOR_BGR2GRAY)
    threshold = 50
    for i in range(foregrounds.shape[0]):
        for j in range(foregrounds.shape[1]):
            if foregrounds[i][j]<threshold:
                foregrounds[i][j]=0
            else:
                foregrounds[i][j]=255          
    plt.imshow(foregrounds,cmap='gray')
    plt.show()