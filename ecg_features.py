# -*- coding: utf-8 -*-
"""
Class to store all features of an ECG epoch.

@author: Pieter
"""

from biosppy.signals import ecg
import numpy as np
from scipy import signal, stats, fftpack
from signalprocessing import findLFHF, dfaecg, dfapiece
from operator import itemgetter


class ECGFeatures:

    def __init__(self, raw_signal, fs=250):
        self.raw_signal = raw_signal
        [_, self.filtered, self.peaks, _, _, heart_rate_ts, self.heart_rate] =\
            ecg.ecg(signal=raw_signal.squeeze(), sampling_rate=fs, show=False)
        self.hrv = heart_rate_ts[1:-1] - heart_rate_ts[0:-2]

        # Normalize Time Series
        self.dethrv = signal.detrend(self.hrv)
        self.nolhrv = stats.zscore(self.hrv)

        # Statistical Feature Extraction
        self.mean = np.mean(self.hrv)
        self.std = np.std(self.hrv)
        self.CV = self.std / self.mean

        # Power spectrum density
        w = np.hamming(len(self.dethrv))
        self.w, self.psd = signal.periodogram(self.dethrv, window=w, detrend=False)
        self.LF, self.HF, self.FreqmaxP, self.maxHFD, self.LFHFratio, self.inter = findLFHF(self.psd, self.w);

        # Detrended Fluctuation Analysis
        self.H, pval95, d, p = dfaecg(self.dethrv)
        self.pval951 = pval95[0]
        self.pval952 = pval95[1]
        if d.size and p.size:
            fitting = np.polyfit(np.log10(d), np.log10(p), 1)
            self.dfaslope = fitting[0]
            self.dfaintercept = fitting[1]
        else:
            self.dfaslope = 0
            self.dfaintercept = 0

        # Theoretical Detrended Fluctuation Analysis
        if d.size and p.size:
            fitting = np.polyfit(np.log10(d), np.log10((np.sqrt(d)) / (np.sqrt(d[0]) / p[0])), 1)
            self.dfaslopeT = fitting[0]
            self.dfainterceptT = fitting[1]
        else:
            self.dfaslopeT = 0
            self.dfainterceptT = 0

        # Piecewise Detrended Fluctuation Analysis
        if len(d) > 1:
            self.alpha1, self.alpha2, self.alpha3, self.alpha1flag, self.alpha2flag, self.alpha3flag = dfapiece(d,p);
        else:
            self.alpha1 = 0
            self.alpha2 = 0
            self.alpha3 = 0
            self.alpha1flag = 0
            self.alpha2flag = 0
            self.alpha3flag = 0

        # Fast Fourier Transformation
        Yfft = fftpack.fft(self.hrv)
        L = len(self.hrv)
        P2 = np.abs(Yfft/L)
        P1 = P2[0:L/2]
        P1[1:-1] = 2*P1[1:-1]
        f = fs * np.array(range(0, (L/2)+1)) / L
        sortIndex, sortedP1 = zip(*sorted(enumerate(P1), reverse=True, key=itemgetter(1)))
        sortedP1 = np.array(sortedP1)

        if sortedP1.size >= 1:
            self.P11 = sortedP1[0]
            self.freq1 = f[sortIndex[0]]
        else:
            self.P11 = 0
            self.freq1 = 0
        if sortedP1.size >= 2:
            self.P12 = sortedP1[1]
            self.freq2 = f[sortIndex[1]]
        else:
            self.P12 = 0
            self.freq2 = 0
        if sortedP1.size >= 3:
            self.P13 = sortedP1[2]
            self.freq3 = f[sortIndex[2]]
        else:
            self.P13 = 0
            self.freq3 = 0
