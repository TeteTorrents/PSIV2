import pickle
from charSegment_pipeline import segmentChars
from feature_extractor import *
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image
import easyocr
import pytesseract

def svm_recognizer(roi):

    # Load models:
    with open('svm_model_numbers.pkl', 'rb') as model_file:
        svm_model_numbers = pickle.load(model_file)

    with open('svm_model_letters.pkl', 'rb') as model_file:
        svm_model_letters = pickle.load(model_file)

    bboxes = segmentChars(roi)
    bboxes_sorted = sorted(bboxes, key=lambda x: x[0])
    result = []

    for idx,bbox in enumerate(bboxes_sorted):
        x_roi, y_roi, w_roi, h_roi = bbox
        char_roi = roi[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
        feature_vector = calculate_feature_vector(char_roi)
        if idx < 4:
            prediction = svm_model_numbers.predict([feature_vector])
        else:
            prediction = svm_model_letters.predict([feature_vector])
        result.append(prediction[0])
    
    return result

def nn_recognizer(roi):
    loaded_model = models.resnet18(pretrained=True)
    loaded_model.fc = nn.Sequential(
        nn.Linear(loaded_model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 36)
    )  
    loaded_model.load_state_dict(torch.load('your_model_state_dict.pth'))

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    bboxes = segmentChars(roi)
    bboxes_sorted = sorted(bboxes, key=lambda x: x[0])
    result = []

    for _,bbox in enumerate(bboxes_sorted):
        x_roi, y_roi, w_roi, h_roi = bbox
        char_roi = roi[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
        char_roi_gray = cv2.cvtColor(char_roi, cv2.COLOR_BGR2GRAY)
        _, aux = cv2.threshold(char_roi_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        inverted_threshed = cv2.bitwise_not(aux)
        pil_image = Image.fromarray(inverted_threshed) 
        preprocessed_image = transform(pil_image).unsqueeze(0) 
        with torch.no_grad():
            loaded_model.eval()
            output = loaded_model(preprocessed_image)
        class_names = [str(i) if i < 10 else chr(i + 55) for i in range(36)]
        _, predicted = output.max(1)
        predicted_letter = class_names[predicted.item()]
        result.append(predicted_letter)
    return result

def easyOCR_recognizer(roi):
    reader = easyocr.Reader(['en'])
    return reader.readtext(roi)[0][1]

def pytesseract_recognizer(roi):
    return pytesseract.image_to_string(roi, config='--psm 8 --oem 3 -c tessedit_char_whitelist=BCDFGHJKLMNPRSTVWXYZ0123456789 --user-patterns xxx.patterns')

if __name__ == '__main__':
    image_o = cv2.imread('fotos/cotxe6.jpg')
    image_o = cv2.resize(image_o, (1000, 800))
    y,x,_ = image_o.shape
    image = image_o[int(1/3*y):, int(1/3*x):]
    x_roi, y_roi, w_roi, h_roi = (195, 166, 289, 83)
    roi = image[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]

    cv2.imshow('Image with Bounding Boxes', roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(svm_recognizer(roi))
    print(nn_recognizer(roi))
    print(easyOCR_recognizer(roi))
    print(pytesseract_recognizer(roi))
