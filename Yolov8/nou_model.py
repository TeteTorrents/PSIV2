from ultralytics import YOLO
import torch

#GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device == 0:
    print("GPU")
    torch.cuda.set_device(device)

print('DEvice = ', device)
model = YOLO('yolov8n.pt')  

results = model.train(data ='./data.yaml', batch = 32, epochs = 450)