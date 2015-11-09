# -*- coding: utf-8 -*-
"""
Evaluates performance of different classifiers.

@author: Pieter
"""

import numpy as np
from sklearn.cross_validation import train_test_split
from data import loadData, groupByEpoch
from feature_construction import dictionaryInitialization, dico_construction
from models import listModels, createNormalizedDataframe
from edr import main

mat_file = 'slpdb.mat'

# Load datas
dataset = loadData(mat_file)
ECG = groupByEpoch(dataset['ECG'])
Resp_in, Resp_out = main(dataset)

# Train dictionary construction
d_train = dictionaryInitialization()
d_train = dico_construction(ECG, Resp_in, Resp_out, d_train)
y = np.array(dataset['binarylabels'])
Xarr_norm = createNormalizedDataframe(d_train)

# Test our models
X_train_train, X_train_test, y_train_train, y_train_test =\
    train_test_split(Xarr_norm, y, test_size=0.2, random_state=42)

X_train_train = np.nan_to_num(X_train_train).reshape((len(X_train_train), 1))
X_train_test = np.nan_to_num(X_train_test).reshape((len(X_train_test), 1))

# Scores
models = listModels()
for m in models:
    print m.__class__.__name__
    print "---------------------------------------"
    print "Params: ", m.get_params()
    m.fit(X_train_train, y_train_train)
    print "Score: ", m.score(X_train_test, y_train_test)
    print
