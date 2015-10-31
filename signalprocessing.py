# -*- coding: utf-8 -*-
"""
Helper functions to process signals.

@author: Pieter Robberechts
"""

from math import floor
import numpy as np
import pandas as pd
from scipy import signal, integrate
import pywt
import warnings
warnings.simplefilter('ignore', np.RankWarning)


'''
-------------------------
-- Frequencies features
-------------------------
'''
# Find the frequency of the signal
def findFreq(serie, fs=200, sel=10):
    fft0 = np.fft.rfft(serie*np.hanning(len(serie)))
    freqs = np.fft.rfftfreq(len(serie), d=1.0/fs)
    fftmod = np.array([np.sqrt(fft0[i].real**2 + fft0[i].imag**2) for i in range(0, len(fft0))])
    d = {'fft': fftmod, 'freq': freqs}
    df = pd.DataFrame(d)
    hop = df.sort(['fft'], ascending=False)
    rows = hop.iloc[:sel]
    return rows['freq'].mean()

def whelchMethod(data, Fs=200):
    f, pxx = signal.welch(data, fs=Fs, nperseg=1024)
    d = {'psd': pxx, 'freqs': f}
    df = pd.DataFrame(data=d)
    dfs = df.sort(['psd'], ascending=False)
    rows = dfs.iloc[:10]
    return rows['freqs'].mean()


def butter_lowpass(cutoff, fs=200, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs=200, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


'''
-----------------------
-- Wavelets features
-----------------------
'''
# Display all the wavelets available
def wavelist():
    for w in pywt.families():
        print pywt.wavelist(w)


# DWT coeffs at level x
def waveCoeffs(data, wave='db4', levels=5):
    w = pywt.wavedec(data, wavelet=wave, level=levels)
    res = list()
    for i in range(1, len(w)):
        temp = 0
        for j in range(0, len(w[i])):
            temp += w[i][j]**2
        res.append(np.sqrt(temp))
    return res


def peakCWT(data):
    pics = signal.find_peaks_cwt(data, widths=np.arange(1,45), wavelet=signal.ricker)
    x = data[pics]
    n = np.linalg.norm(x)
    return n


'''
-----------------------
-- Signal features
-----------------------
'''
def autoCorrelation(data):
    return signal.correlate(data,data).mean()


def findLFHF(psd, w):
    VLFpsd = VLFw = LFpsd = LFw = HFpsd = HFw = np.empty(0)
    m = w.shape[0]

    for i in range(0, m):
        if w[i] <= 0.05:
            VLFpsd = np.append(VLFpsd, psd[i])
            VLFw = np.append(VLFw, w[i])
        if w[i] > 0.05 and w[i] <= 0.15:
            LFpsd = np.append(LFpsd, psd[i])
            LFw = np.append(LFw, w[i])
        if w[i] > 0.15 and w[i] <= 0.4:
            HFpsd = np.append(HFpsd, psd[i])
            HFw = np.append(HFw, w[i])

    LF = integrate.trapz(LFw, LFpsd) / (integrate.trapz(w, psd) - integrate.trapz(VLFw, VLFpsd))
    HF = integrate.trapz(HFw, HFpsd) / (integrate.trapz(w, psd) - integrate.trapz(VLFw, VLFpsd))
    LFHFratio = LF / HF
    inter = LF / (LF + HF)
    if HFpsd.size:
        [maxHFD, maxIndex] = max((v, i) for i, v in enumerate(HFpsd))
        FreqmaxP = HFw[maxIndex]
    else:
        maxHFD = 0
        FreqmaxP = 0
    return (LF, HF, FreqmaxP, maxHFD, LFHFratio, inter)


def dfaecg(x, d=10):
    """DFA Calculate the Hurst exponent using DFA analysis.

    H = DFA(X) calculates the Hurst exponent of time series X using
    Detrended Fluctuation Analysis (DFA). If a vector of increasing natural
    numbers is given as the second input parameter, i.e. DFA(X,D), then it
    defines the box sizes that the sample is divided into (the values in D
    have to be divisors of the length of series X). If D is a scalar
    (default value D = 10) it is treated as the smallest box size that the
    sample can be divided into. In this case the optimal sample size OptN
    and the vector of divisors for this size are automatically computed.
    OptN is defined as the length that possesses the most divisors among
    series shorter than X by no more than 1%. The input series X is
    truncated at the OptN-th value.
    [H,PV95] = DFA(X) returns the empirical 95% confidence intervals PV95
    (see [2]).
    [H,PV95,P] = DFA(X) returns the average standard deviations P of the
    detrended walk for all the divisors.

    If there are no output parameters, the DFA statistics is automatically
    plotted against the divisors on a loglog paper and the results of the
    analysis are displayed in the command window. DFA(X,D,FONTSIZE) allows
    to specify a fontsize different than 14 in the plotted figure.

    References:
    [1] C.-K.Peng et al. (1994) Mosaic organization of DNA nucleotides,
    Physical Review E 49(2), 1685-1689.
    [2] R.Weron (2002) Estimating long range dependence: finite sample
    properties and confidence intervals, Physica A 312, 285-299.

    Translated from Matlab code written by Rafal Weron (2011.09.30).
    Based on functions dfa.m, dfacalc.m, finddiv.m, findndiv.m originally
    written by Beata Przyby?wicz & Rafal Weron (2000.06.30, 2000.12.15,
    2002.07.27).
    """
    if not isinstance(d, list):
        # For scalar d set dmin=d and find the 'optimal' vector d
        dmin = d
        # Find such a natural number OptN that possesses the largest number of
        # divisors among all natural numbers in the interval [0.99*N,N]
        N = len(x)
        N0 = int(floor(0.99*N))
        dv = np.zeros(shape=(N-N0+1, 1))
        for i in range(N0, N+1):
            dv[i-N0] = len(divisors(i, dmin))
        OptN = N0 + dv[np.where(max(dv) == dv)] - 1

        # Use the first OptN values of x for further analysis
        OptN = OptN[0]
        x = x[0:OptN]
        # Find the divisors of x
        d = divisors(OptN, dmin)
    else:
        OptN = len(x)

    # Construct a 'random walk' out of the return time series x and calculate
    # the root mean square fluctuation (i.e. standard deviation) of the
    # integrated and detrended time series (see p.288 in [2])
    p = np.zeros(shape=(len(d), 1))
    y = np.cumsum(x)
    for i in range(0, len(d)):
        p[i] = RMSfluctuation(y, d[i])

    # Compute the Hurst exponent as the slope on a loglog scale
    try:
        pp = np.polyfit(np.log10(d), np.log10(p), 1)
    except:
        pp = [0, 0]
    H = pp[0]

    # Compute empirical confidence intervals (see [2])
    L = np.log2(OptN)
    if dmin > 50:
        # DFA (min(divisor)>50) two-sided empirical confidence intervals
        pval95 = [0.5-np.exp(-2.93*np.log(L)+4.45), np.exp(-3.10*np.log(L)+4.77)+0.5]
        C = [0.5-np.exp(-2.99*np.log(L)+4.45), np.exp(-3.09*np.log(L)+4.57)+0.5, .90]
        C.append([pval95, .95])
        C.append([0.5-np.exp(-2.67*np.log(L)+4.06), np.exp(-3.19*np.log(L)+5.28)+0.5, .99])
    else:
        # DFA (min(divisor)>10) two-sided empirical confidence intervals
        pval95 = [0.5-np.exp(-2.33*np.log(L)+3.25), np.exp(-2.46*np.log(L)+3.38)+0.5]
        C = [0.5-np.exp(-2.33*np.log(L)+3.09), np.exp(-2.44*np.log(L)+3.13)+0.5, .90]
        C.append([pval95, .95])
        C.append([0.5-np.exp(-2.20*np.log(L)+3.18), np.exp(-2.45*np.log(L)+3.62)+0.5, .99])

    return (H, pval95, d, p)


def divisors(n, n0):
    """Find all divisors of the natural number N greater or equal to N0
    """
    i = np.array(range(n0, int(floor(n/2)) + 1))
    return i[np.where((float(n)/i) == np.floor(float(n)/i))]


def RMSfluctuation(x, d):
    """Calculate the root mean square fluctuation
    """
    n = len(x)/d
    X = np.reshape(x, (d, n))
    Y = X
    t = np.array(range(1, d+1)).T
    for i in range(0, n):
        p = np.polyfit(t, X[:,i], 1)
        Y[:,i] = X[:,i] - t*p[0] - p[1]
    return np.mean(np.std(Y))


def dfapiece(d, p):
    """Piecewise Detrended Fluctuation Analysis

    https://en.wikipedia.org/wiki/Detrended_fluctuation_analysis
    """
    m = len(d)
    tempY1 = tempX1 = tempY2 = tempX2 = tempY3 = tempX3 = np.zeros(0)

    for i in range(0, m):
        if (d[i] >= 10) and (d[i] <= 30):
            tempY1 = np.append(tempY1, p[i])
            tempX1 = np.append(tempX1, d[i])
        if (d[i] >= 30) and (d[i] <= 300):
            tempY2 = np.append(tempY2, p[i])
            tempX2 = np.append(tempX2, d[i])
        if (d[i] >= 300):
            tempY3 = np.append(tempY3, p[i])
            tempX3 = np.append(tempX3, d[i])

    if len(tempX1) > 1:
        alpha1 = np.polyfit(np.log10(tempX1), np.log10(tempY1), 1)
        alpha1 = alpha1[0]
        alpha1flag = 1
    else:
        alpha1 = 0
        alpha1flag = 0

    if len(tempX2) > 1:
        alpha2 = np.polyfit(np.log10(tempX2), np.log10(tempY2), 1)
        alpha2 = alpha2[0]
        alpha2flag = 1
    else:
        alpha2 = 0
        alpha2flag = 0

    if len(tempX3) > 1:
        alpha3 = np.polyfit(np.log10(tempX3), np.log10(tempY3), 1)
        alpha3 = alpha3[0]
        alpha3flag = 1
    else:
        alpha3 = 0
        alpha3flag = 0

    return (alpha1, alpha2, alpha3, alpha1flag, alpha2flag, alpha3flag)
