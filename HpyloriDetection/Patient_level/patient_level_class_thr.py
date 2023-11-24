import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
import random
from sklearn.preprocessing import LabelEncoder
from scipy import stats

chunk_size = 52 
def random_chunk(lst, chunk_size):
    start = random.randint(0, len(lst) - chunk_size)
    return lst[start:start + chunk_size]

gt_fold1 = ['POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE']
pred_fold1 = [0.2719745222929936, 0.18340306834030684, 0.22407307898979043, 0.008403361344537815, 0.0063099444724886425, 0.10586944596818432, 0.0, 0.002297090352220521, 0.19144013880855987, 0.009208103130755065, 0.004149377593360996, 0.0004065040650406504, 0.11341681574239713, 0.05192878338278932, 0.0759493670886076, 0.0, 0.0, 0.017857142857142856, 0.026535253980288095, 0.0009514747859181732, 0.0, 0.1064773735581189, 0.0024875621890547263, 0.004933051444679351, 0.01288659793814433, 0.009202453987730062, 0.03005181347150259, 0.0038535645472061657, 0.005008347245409015, 0.39537869062901154, 0.5601173020527859, 0.329938900203666, 0.07102728731942215, 0.0039919243828576674, 0.033946251768033946, 0.11925866236905722, 0.00033783783783783786, 0.0842443729903537, 0.00011229646266142616, 0.08970886932972241, 0.07490272373540856, 0.22952529994783516, 0.02875924404272802, 0.04064272211720227, 0.07407407407407407, 0.005274261603375527, 0.04738154613466334, 0.05333333333333334, 0.014846235418875928, 0.02714535901926445, 0.0, 0.0, 0.14235500878734622, 0.0014492753623188406, 0.06650717703349282, 0.08314606741573034, 0.1952191235059761, 0.14727011494252873, 0.22582181991424488, 0.012946979038224414, 0.0, 0.009194395796847636, 0.46630727762803237, 0.01310615989515072, 0.0, 0.02396021699819168, 0.004755434782608696, 0.3078034682080925, 0.05511346890657236, 0.0064464141821112, 0.031073446327683617, 0.04022346368715084, 0.20585585585585586, 0.0, 0.005338078291814947, 0.0035881865857024567, 0.0005672149744753262, 0.009846690764053347, 0.0, 0.11780104712041885, 0.009993753903810118, 0.14054927302100162, 0.4929317762753534, 0.0017709563164108619, 0.005843071786310518, 0.010232796111537478, 0.00031308703819661864, 0.5451895043731778, 0.6060606060606061, 0.0, 0.019417475728155338, 0.018746652383502947, 0.014939759036144579, 0.0012254901960784314, 0.07897934386391252, 0.025036818851251842, 0.0, 0.01791304347826087, 0.16859122401847576, 0.001736111111111111, 0.0, 0.001053740779768177, 0.20841683366733466, 0.013598326359832637, 0.14933333333333335, 0.05120056497175141, 0.003933910306845004, 0.30179171332586785, 0.0, 0.0008144980655670943, 0.0035, 0.0005497526113249038, 0.03543913713405239, 0.4879356568364611, 0.12653721682847896, 0.003155479059093517, 0.035924932975871314, 0.0, 0.12141372141372142, 0.10289389067524116, 0.6963064295485636, 0.0, 0.0009017132551848512, 0.0024154589371980675, 0.17024041585445093, 0.0005704506560182544, 0.0, 0.06179286335944299, 0.0002760905577029266, 0.15597235932872655, 0.026066350710900472, 0.0010416666666666667, 0.0, 0.0, 0.0, 0.011698757007068, 0.17490729295426452, 0.013716814159292035, 0.001176470588235294, 0.08041958041958042, 0.06993511175198269, 0.00021413276231263382, 0.0, 0.10392064241851677, 0.028506271379703536, 0.0004288164665523156, 0.0025560920193126954, 0.036968576709796676, 0.13365539452495975, 0.0031821797931583136, 0.0, 0.001443001443001443, 0.14691558441558442, 0.056896551724137934, 0.0008748906386701663, 0.02225519287833828]
gt_fold2 = ['NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE']
pred_fold2 = [0.0054611650485436895, 0.2712345432432426, 0.008403361344537815, 0.0063099444724886425, 0.10586944596818432, 0.0, 0.0022701475595913734, 0.3211436170212766, 0.002297090352220521, 0.009208103130755065, 0.004149377593360996, 0.11341681574239713, 0.05192878338278932, 0.0, 0.0, 0.017857142857142856, 0.005658852061438965, 0.026535253980288095, 0.0009514747859181732, 0.1064773735581189, 0.29193842942864595, 0.0002492522432701894, 0.0024875621890547263, 0.004933051444679351, 0.01288659793814433, 0.009202453987730062, 0.03005181347150259, 0.0038535645472061657, 0.2745664739884393, 0.39537869062901154, 0.5601173020527859, 0.07102728731942215, 0.0039919243828576674, 0.013844515441959531, 0.033946251768033946, 0.11925866236905722, 0.00033783783783783786, 0.00011229646266142616, 0.08970886932972241, 0.07490272373540856, 0.22952529994783516, 0.07667731629392971, 0.02875924404272802, 0.04064272211720227, 0.005274261603375527, 0.05333333333333334, 0.6567219152854512, 0.014846235418875928, 0.02714535901926445, 0.1841876629018245, 0.002386634844868735, 0.041527126590756865, 0.08955223880597014, 0.0, 0.14235500878734622, 0.002442002442002442, 0.06386554621848739, 0.06650717703349282, 0.08314606741573034, 0.1952191235059761, 0.08573853989813243, 0.0, 0.22582181991424488, 0.05911823647294589, 0.0, 0.14054927302100162, 0.08148517318714178, 0.009194395796847636, 0.17861975642760486, 0.46630727762803237, 0.01310615989515072, 0.0, 0.02396021699819168, 0.004755434782608696, 0.3078034682080925, 0.0064464141821112, 0.04022346368715084, 0.0, 0.005338078291814947, 0.0005672149744753262, 0.006585287504015419, 0.0025680534155110425, 0.0, 0.11780104712041885, 0.009993753903810118, 0.14054927302100162, 0.4929317762753534, 0.0017709563164108619, 0.0, 0.005762581636573185, 0.005843071786310518, 0.5451895043731778, 0.0, 0.019417475728155338, 0.009009009009009009, 0.0009770395701025891, 0.018746652383502947, 0.014939759036144579, 0.0055178268251273345, 0.32270916334661354, 0.07897934386391252, 0.025036818851251842, 0.0, 0.08328479906814211, 0.0, 0.01791304347826087, 0.16859122401847576, 0.0, 0.001053740779768177, 0.002170767004341534, 0.013598326359832637, 0.14933333333333335, 0.05120056497175141, 0.003933910306845004, 0.30179171332586785, 0.0, 0.009625668449197862, 0.0035, 0.03543913713405239, 0.4879356568364611, 0.12653721682847896, 0.035924932975871314, 0.0, 0.005619556055071649, 0.07428571428571429, 0.6963064295485636, 0.0, 0.0009017132551848512, 0.0024154589371980675, 0.0005704506560182544, 0.0, 0.0002760905577029266, 0.15597235932872655, 0.001802451333813987, 0.0010416666666666667, 0.0, 0.0, 0.011698757007068, 0.0010723860589812334, 0.17490729295426452, 0.013716814159292035, 0.003245248029670839, 0.001176470588235294, 0.08041958041958042, 0.06993511175198269, 0.00021413276231263382, 0.0, 0.10392064241851677, 0.028506271379703536, 0.0004288164665523156, 0.0025560920193126954, 0.0, 0.001443001443001443, 0.14691558441558442, 0.056896551724137934]
gt_fold3 = ['NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE']
pred_fold3 = [0.005441550248552345, 0.297422422929936, 0.18340306834030684, 0.22407307898979043, 0.0063099444724886425, 0.10586944596818432, 0.0, 0.0022701475595913734, 0.3211436170212766, 0.002297090352220521, 0.19144013880855987, 0.009208103130755065, 0.004149377593360996, 0.0004065040650406504, 0.11341681574239713, 0.0759493670886076, 0.0, 0.0, 0.017857142857142856, 0.005658852061438965, 0.0009514747859181732, 0.0, 0.1064773735581189, 0.29193842942864595, 0.0002492522432701894, 0.0024875621890547263, 0.01288659793814433, 0.03005181347150259, 0.0038535645472061657, 0.005008347245409015, 0.2745664739884393, 0.39537869062901154, 0.5601173020527859, 0.329938900203666, 0.013844515441959531, 0.11925866236905722, 0.00033783783783783786, 0.0842443729903537, 0.00011229646266142616, 0.08970886932972241, 0.07490272373540856, 0.22952529994783516, 0.07667731629392971, 0.02875924404272802, 0.07407407407407407, 0.005274261603375527, 0.04738154613466334, 0.6567219152854512, 0.014846235418875928, 0.02714535901926445, 0.1841876629018245, 0.002386634844868735, 0.041527126590756865, 0.08955223880597014, 0.0, 0.0, 0.002442002442002442, 0.06386554621848739, 0.0014492753623188406, 0.06650717703349282, 0.1952191235059761, 0.08573853989813243, 0.14727011494252873, 0.0, 0.012946979038224414, 0.05911823647294589, 0.0, 0.14054927302100162, 0.08148517318714178, 0.17861975642760486, 0.0, 0.004755434782608696, 0.3078034682080925, 0.05511346890657236, 0.031073446327683617, 0.20585585585585586, 0.0, 0.005338078291814947, 0.0035881865857024567, 0.0005672149744753262, 0.006585287504015419, 0.0025680534155110425, 0.009846690764053347, 0.11780104712041885, 0.14054927302100162, 0.4929317762753534, 0.0017709563164108619, 0.0, 0.005762581636573185, 0.010232796111537478, 0.00031308703819661864, 0.5451895043731778, 0.6060606060606061, 0.0, 0.019417475728155338, 0.009009009009009009, 0.0009770395701025891, 0.014939759036144579, 0.0012254901960784314, 0.0055178268251273345, 0.32270916334661354, 0.0, 0.08328479906814211, 0.16859122401847576, 0.001736111111111111, 0.001053740779768177, 0.002170767004341534, 0.20841683366733466, 0.05120056497175141, 0.003933910306845004, 0.0, 0.0008144980655670943, 0.009625668449197862, 0.0035, 0.0005497526113249038, 0.03543913713405239, 0.4879356568364611, 0.12653721682847896, 0.003155479059093517, 0.035924932975871314, 0.005619556055071649, 0.07428571428571429, 0.12141372141372142, 0.10289389067524116, 0.0, 0.0024154589371980675, 0.17024041585445093, 0.0005704506560182544, 0.0, 0.06179286335944299, 0.0002760905577029266, 0.15597235932872655, 0.026066350710900472, 0.001802451333813987, 0.0010416666666666667, 0.0, 0.0, 0.0, 0.0010723860589812334, 0.17490729295426452, 0.013716814159292035, 0.003245248029670839, 0.001176470588235294, 0.08041958041958042, 0.06993511175198269, 0.00021413276231263382, 0.0, 0.10392064241851677, 0.028506271379703536, 0.0004288164665523156, 0.036968576709796676, 0.13365539452495975, 0.0031821797931583136, 0.001443001443001443, 0.0008748906386701663, 0.02225519287833828]
gt_fold4 = ['NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE']
pred_fold4 = [0.0, 0.11780104712041885, 0.009993753903810118, 0.14054927302100162, 0.4929317762753534, 0.0017709563164108619, 0.005843071786310518, 0.010232796111537478, 0.00031308703819661864, 0.5451895043731778, 0.6060606060606061, 0.0, 0.019417475728155338, 0.018746652383502947, 0.014939759036144579, 0.0012254901960784314, 0.07897934386391252, 0.025036818851251842, 0.0, 0.01791304347826087, 0.16859122401847576, 0.001736111111111111, 0.0, 0.001053740779768177, 0.20841683366733466, 0.013598326359832637, 0.14933333333333335, 0.05120056497175141, 0.003933910306845004, 0.30179171332586785, 0.0, 0.0008144980655670943, 0.0035, 0.0005497526113249038, 0.03543913713405239, 0.4879356568364611, 0.12653721682847896, 0.003155479059093517, 0.035924932975871314, 0.0, 0.12141372141372142, 0.10289389067524116, 0.6963064295485636, 0.0, 0.0009017132551848512, 0.0024154589371980675, 0.17024041585445093, 0.0005704506560182544, 0.0, 0.06179286335944299, 0.0002760905577029266, 0.15597235932872655, 0.026066350710900472, 0.0010416666666666667, 0.0, 0.0, 0.0, 0.011698757007068, 0.17490729295426452, 0.013716814159292035, 0.001176470588235294, 0.08041958041958042, 0.06993511175198269, 0.00021413276231263382, 0.0, 0.10392064241851677, 0.028506271379703536, 0.0004288164665523156, 0.0025560920193126954, 0.036968576709796676, 0.13365539452495975, 0.0031821797931583136, 0.0, 0.001443001443001443, 0.14691558441558442, 0.056896551724137934, 0.0008748906386701663, 0.02225519287833828, 0.0054611650485436895, 0.2719745222929936, 0.008403361344537815, 0.0063099444724886425, 0.10586944596818432, 0.0, 0.0022701475595913734, 0.3211436170212766, 0.002297090352220521, 0.009208103130755065, 0.004149377593360996, 0.11341681574239713, 0.05192878338278932, 0.0, 0.0, 0.017857142857142856, 0.005658852061438965, 0.026535253980288095, 0.0009514747859181732, 0.1064773735581189, 0.29193842942864595, 0.0002492522432701894, 0.0024875621890547263, 0.004933051444679351, 0.01288659793814433, 0.009202453987730062, 0.03005181347150259, 0.0038535645472061657, 0.2745664739884393, 0.39537869062901154, 0.5601173020527859, 0.07102728731942215, 0.0039919243828576674, 0.013844515441959531, 0.033946251768033946, 0.11925866236905722, 0.00033783783783783786, 0.00011229646266142616, 0.08970886932972241, 0.07490272373540856, 0.22952529994783516, 0.07667731629392971, 0.02875924404272802, 0.04064272211720227, 0.005274261603375527, 0.05333333333333334, 0.6567219152854512, 0.014846235418875928, 0.02714535901926445, 0.1841876629018245, 0.002386634844868735, 0.041527126590756865, 0.08955223880597014, 0.0, 0.14235500878734622, 0.002442002442002442, 0.06386554621848739, 0.06650717703349282, 0.08314606741573034, 0.1952191235059761, 0.08573853989813243, 0.0, 0.22582181991424488, 0.05911823647294589, 0.0, 0.14054927302100162, 0.08148517318714178, 0.009194395796847636, 0.17861975642760486, 0.46630727762803237, 0.01310615989515072, 0.0, 0.02396021699819168, 0.004755434782608696, 0.3078034682080925, 0.0064464141821112, 0.04022346368715084, 0.0]
gt_fold5 = ['NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE', 'NEGATIVE']
pred_fold5 = [0.005338078291814947, 0.0005672149744753262, 0.006585287504015419, 0.0025680534155110425, 0.0, 0.11780104712041885, 0.009993753903810118, 0.14054927302100162, 0.4929317762753534, 0.0017709563164108619, 0.0, 0.005762581636573185, 0.005843071786310518, 0.5451895043731778, 0.0, 0.019417475728155338, 0.009009009009009009, 0.0009770395701025891, 0.018746652383502947, 0.014939759036144579, 0.0055178268251273345, 0.32270916334661354, 0.07897934386391252, 0.025036818851251842, 0.0, 0.08328479906814211, 0.0, 0.01791304347826087, 0.16859122401847576, 0.0, 0.001053740779768177, 0.002170767004341534, 0.013598326359832637, 0.14933333333333335, 0.05120056497175141, 0.003933910306845004, 0.30179171332586785, 0.0, 0.009625668449197862, 0.0035, 0.03543913713405239, 0.4879356568364611, 0.12653721682847896, 0.035924932975871314, 0.0, 0.005619556055071649, 0.07428571428571429, 0.6963064295485636, 0.0, 0.0009017132551848512, 0.0024154589371980675, 0.0005704506560182544, 0.0, 0.0002760905577029266, 0.15597235932872655, 0.001802451333813987, 0.0010416666666666667, 0.0, 0.0, 0.011698757007068, 0.0010723860589812334, 0.17490729295426452, 0.013716814159292035, 0.003245248029670839, 0.001176470588235294, 0.08041958041958042, 0.06993511175198269, 0.00021413276231263382, 0.0, 0.10392064241851677, 0.028506271379703536, 0.0004288164665523156, 0.0025560920193126954, 0.0, 0.001443001443001443, 0.14691558441558442, 0.056896551724137934, 0.2719745222929936, 0.18340306834030684, 0.22407307898979043, 0.008403361344537815, 0.0063099444724886425, 0.10586944596818432, 0.0, 0.002297090352220521, 0.19144013880855987, 0.009208103130755065, 0.004149377593360996, 0.0004065040650406504, 0.11341681574239713, 0.05192878338278932, 0.0759493670886076, 0.0, 0.0, 0.017857142857142856, 0.026535253980288095, 0.0009514747859181732, 0.0, 0.1064773735581189, 0.0024875621890547263, 0.004933051444679351, 0.01288659793814433, 0.009202453987730062, 0.03005181347150259, 0.0038535645472061657, 0.005008347245409015, 0.39537869062901154, 0.5601173020527859, 0.329938900203666, 0.07102728731942215, 0.0039919243828576674, 0.033946251768033946, 0.11925866236905722, 0.00033783783783783786, 0.0842443729903537, 0.00011229646266142616, 0.08970886932972241, 0.07490272373540856, 0.22952529994783516, 0.02875924404272802, 0.04064272211720227, 0.07407407407407407, 0.005274261603375527, 0.04738154613466334, 0.05333333333333334, 0.014846235418875928, 0.02714535901926445, 0.0, 0.0, 0.14235500878734622, 0.0014492753623188406, 0.06650717703349282, 0.08314606741573034, 0.1952191235059761, 0.14727011494252873, 0.22582181991424488, 0.012946979038224414, 0.0, 0.009194395796847636, 0.46630727762803237, 0.01310615989515072, 0.0, 0.02396021699819168, 0.004755434782608696, 0.3078034682080925, 0.05511346890657236, 0.0064464141821112, 0.031073446327683617, 0.04022346368715084, 0.20585585585585586, 0.0, 0.005338078291814947, 0.0035881865857024567, 0.0005672149744753262, 0.009846690764053347]

folds = [(gt_fold1, pred_fold1), (gt_fold2, pred_fold2), (gt_fold3, pred_fold3), (gt_fold4, pred_fold4), (gt_fold5, pred_fold5)]

label_encoder = LabelEncoder()

def calculate_mean_roc(folds):
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)
    for gt, pred in folds:
        gt_binary = label_encoder.fit_transform(gt)
        fpr, tpr, _ = roc_curve(gt_binary, pred)
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr /= len(folds)

    mean_roc_auc = auc(mean_fpr, mean_tpr)

    return mean_fpr, mean_tpr, mean_roc_auc

def calculate_threshold_and_margin(folds):
    thresholds_all = []

    for gt, pred in folds:
        gt_binary = label_encoder.fit_transform(gt)
        fpr, tpr, thresholds = roc_curve(gt_binary, pred)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        thresholds_all.append(optimal_threshold)

    mean_threshold = np.mean(thresholds_all)
    std_threshold = np.std(thresholds_all, ddof=1) 
    confidence_level = 0.95
    margin_of_error = stats.t.ppf((1 + confidence_level) / 2, len(thresholds_all) - 1) * (std_threshold / np.sqrt(len(thresholds_all)))

    return mean_threshold, margin_of_error

for idx, (gt, pred) in enumerate(folds, start=1):
    gt_binary = label_encoder.fit_transform(gt)
    fpr, tpr, thresholds = roc_curve(gt_binary, pred)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for each fold with alpha=0.6
    plt.plot(fpr, tpr, label=f'Fold {idx} (AUC = {roc_auc:.2f})', alpha=0.6)

mean_fpr, mean_tpr, mean_roc_auc = calculate_mean_roc(folds)
plt.plot(mean_fpr, mean_tpr, label=f'Mean ROC (AUC = {mean_roc_auc:.2f})', color='b', linewidth=2)

mean_threshold, margin_of_error = calculate_threshold_and_margin(folds)
plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.scatter(mean_fpr[np.argmax(mean_tpr - mean_fpr)], mean_tpr[np.argmax(mean_tpr - mean_fpr)], marker='o', color='r', label=f'Mean Threshold ({mean_threshold:.5f} \u00B1 {margin_of_error:.5f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve for mean of All 5 Folds and Mean ROC Curve')
plt.legend(loc='lower right')
plt.show()

print("DONE")

