# -*- coding: utf-8 -*-
"""
sklearn classification models.

@author: Pieter
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

'''
-----------------------
-- Models
-----------------------
'''
def listModels():
    models = list()
    models.append(KNeighborsClassifier(n_neighbors=20))
    models.append(LogisticRegression(penalty='l1'))
    models.append(AdaBoostClassifier())
    models.append(GradientBoostingClassifier())
    models.append(RandomForestClassifier())
    models.append(LDA())
    models.append(DecisionTreeClassifier())
    models.append(SVC(kernel='linear'))
    return models

'''
-----------------------
-- Useful
-----------------------
'''
def savePredictions(pred_res,filename='y_pred.txt'):
    np.savetxt(filename, pred_res, fmt='%s')


def createNormalizedDataframe(dic):
    df = pd.DataFrame(data=dic)
    df_norm = (df - df.mean()) / df.std()
    return np.array(df_norm.to_records(), dtype=np.float64)
