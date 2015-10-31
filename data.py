# -*- coding: utf-8 -*-
"""
Functions to load and store the data.

@author: Pieter
"""

import numpy as np
from scipy import io


'''
---------------------------------------------------
--- Load data
---------------------------------------------------
'''
def loadData(filename='data.mat'):
    return io.loadmat(filename, squeeze_me=True)


def groupByEpoch(signal, fs=250, epoch_length=30):
    N = fs * epoch_length
    return [signal[n:n+N] for n in range(0, len(signal), N)]

'''
---------------------------------------------------
--- Save / Load python numpy arrays
---------------------------------------------------
'''
def saveBinaryMatrix(X, filename='datax.bin'):
    f = file(filename, 'wb')
    np.save(f, X)
    f.close()


def loadBinaryMatrix(filename='dataX.bin'):
    f = file(filename, 'rb')
    X = np.load(f)
    f.close()
    return X
