# -*- coding: utf-8 -*-
"""
Train a model.

@author: Pieter
"""

import os
import numpy as np
from data import loadData, groupByEpoch
from models import createNormalizedDataframe
from featureConstruction import dictionaryInitialization, dico_construction
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib


# Load datas
ECG = np.array([])
labels = np.array([])
binarylabels = np.array([])
for file in os.listdir("traindata"):
    if file.endswith(".mat"):
        dataset = loadData('traindata/' + file)
        ECG = np.append(ECG, dataset['ECG'])
        labels = np.append(labels, dataset['labels'])
        binarylabels = np.append(binarylabels, dataset['binarylabels'])
X = groupByEpoch(ECG)

# Train dictionary construction
d_train = dictionaryInitialization()
d_train = dico_construction(X, d_train)
y = np.array(labels)
Xarr_norm = createNormalizedDataframe(d_train)
Xarr_norm = np.nan_to_num(Xarr_norm).reshape((len(Xarr_norm), 1))

# Compute model and write to file
model = DecisionTreeClassifier().fit(Xarr_norm, y)
joblib.dump(model, 'model.pkl')
