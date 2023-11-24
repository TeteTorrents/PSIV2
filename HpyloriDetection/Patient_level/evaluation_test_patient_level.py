import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

gt = ['NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE']
pred = [0.008403361344537815, 0.05192878338278932, 0.026535253980288095, 0.004933051444679351, 0.009202453987730062, 0.07102728731942215, 0.0039919243828576674, 0.033946251768033946, 0.04064272211720227, 0.05333333333333334, 0.14235500878734622, 0.08314606741573034, 0.22582181991424488, 0.009194395796847636, 0.46630727762803237, 0.01310615989515072, 0.02396021699819168, 0.0064464141821112, 0.04022346368715084, 0.0, 0.009993753903810118, 0.005843071786310518, 0.018746652383502947, 0.07897934386391252, 0.025036818851251842, 0.0, 0.01791304347826087, 0.0, 0.013598326359832637, 0.14933333333333335, 0.30179171332586785, 0.0, 0.6963064295485636, 0.0009017132551848512, 0.011698757007068, 0.0025560920193126954, 0.0, 0.14691558441558442, 0.056896551724137934, 0.18340306834030684, 0.22407307898979043, 0.19144013880855987, 0.0004065040650406504, 0.0759493670886076, 0.0, 0.005008347245409015, 0.329938900203666, 0.0842443729903537, 0.07407407407407407, 0.04738154613466334, 0.0, 0.0014492753623188406, 0.14727011494252873, 0.012946979038224414, 0.05511346890657236, 0.031073446327683617, 0.20585585585585586, 0.0035881865857024567, 0.009846690764053347, 0.010232796111537478, 0.00031308703819661864, 0.6060606060606061, 0.0012254901960784314, 0.001736111111111111, 0.20841683366733466, 0.0008144980655670943, 0.0005497526113249038, 0.003155479059093517, 0.12141372141372142, 0.10289389067524116, 0.17024041585445093, 0.06179286335944299, 0.026066350710900472, 0.0, 0.036968576709796676, 0.13365539452495975, 0.0031821797931583136, 0.0008748906386701663, 0.02225519287833828]

threshold = 0.02256

binary_predictions = ['POSITIVE' if p >= threshold else 'NEGATIVE' for p in pred]

conf_matrix = confusion_matrix(gt, binary_predictions, labels=['NEGATIVE', 'POSITIVE'])
precision = precision_score(gt, binary_predictions, pos_label='POSITIVE')
recall = recall_score(gt, binary_predictions, pos_label='POSITIVE')
f1 = f1_score(gt, binary_predictions, pos_label='POSITIVE')

accuracy = accuracy_score(gt, binary_predictions)

print("Confusion Matrix:")
print(conf_matrix)
print("\nPrecision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("Accuracy:", accuracy)

gt_binary = np.array([1 if label == 'POSITIVE' else 0 for label in gt])
pred_positive = [pred[i] for i in range(len(pred)) if gt_binary[i] == 1]
pred_negative = [pred[i] for i in range(len(pred)) if gt_binary[i] == 0]


data = {
    'POSITIVE': [val * 100 for val in pred_positive],
    'NEGATIVE': [val * 100 for val in pred_negative]
}

fig, ax = plt.subplots()
boxplot = ax.boxplot([data['POSITIVE'], data['NEGATIVE']], labels=['POSITIVE', 'NEGATIVE'])
threshold_line = ax.axhline(y=2.256, color='black', linestyle='--', label='thr = 2.256%')
plt.title('Boxplots for POSITIVE and NEGATIVE Classes')
plt.ylabel('Percentage (%)')
plt.legend()
plt.show()

print("DONE")