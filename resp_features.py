# -*- coding: utf-8 -*-
"""
Class to store all features of an ECG epoch.

@author: Pieter
"""

import numpy as np


class RespFeatures:

    def __init__(self, signal, fs=250):
        self.signal = signal

        # Statistical Feature Extraction
        self.mean = np.mean(self.signal)
        self.std = np.std(self.signal)
        self.CV = self.std / self.mean
