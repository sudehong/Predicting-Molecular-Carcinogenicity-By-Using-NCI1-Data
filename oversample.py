import numpy as np
import torch
np.random.seed(0)
torch.manual_seed(0)
import pandas as pd
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.drawOptions.addAtomIndices = True
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from uilt1 import MyModel
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix,r2_score
import torch.nn.functional as F

from imblearn.over_sampling import SMOTE

from collections import Counter
from imblearn.over_sampling import RandomOverSampler

data = pd.read_csv("C:\\Users\\SU DEHONG\\Desktop\\NCIi\\NCI1_dataset_2.csv")
y = data.iloc[:, 1].values
X = data.iloc[:, 0].values

i = np.array([i for i  in range(data.shape[0])])

# le = LabelEncoder()
# label = le.fit_transform(y)

# Counter(y)
# print(i.shape,type(i),y.shape)


# smo = SMOTE()
# i_smo, y_smo = smo.fit_resample(i.reshape(-1, 1), label)


ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X.reshape(-1, 1), y)

# X_resampled.squeeze().shape
# y_resampled.shape

# Counter(label)
# for i in i_smo:
# #     print(i)
data = {'smile':X_resampled.squeeze(),'activity':y_resampled}
df = pd.DataFrame(data)
print(df)
df.to_csv('C:\\Users\\SU DEHONG\\Desktop\\NCIi\\NCI1_dataset_oversample.csv')
