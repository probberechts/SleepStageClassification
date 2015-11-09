# -*- coding: utf-8 -*-
"""
Derive a respiration signal from an ECG
Based on: http://www.physionet.org/physiotools/edr/edr.c
@author: Pieter
"""

import numpy as np
from math import ceil
from data import loadData
from scipy import interpolate
from ecg_features import ECGFeatures
from biosppy.signals import resp


def median(lst):
    return np.median(np.array(lst))

def detectQRSPeaks(ecg, debug=False):
    rpeaks = ECGFeatures(ecg).peaks

    if (debug is True):
        import matplotlib.pyplot as plt
        plt.plot(ecg, 'b', rpeaks, ecg[rpeaks], 'r*')
        plt.show()

    return rpeaks


def detectQRSOnset(ecg, debug=False):
    time = 0
    now = 10

    freq = 250
    ms160 = ceil(0.16*freq)
    ms200 = ceil(0.2*freq)
    s2 = ceil(2*freq)
    scmin = 0.4
    # number of ADC units corresponding to scmin microvolts
    # scmin = muvadu(signal, scmin);
    scmax = 10 * scmin
    slopecrit = 10 * scmin
    maxslope = 0
    nslope = 0
    out = []

    while (now < len(ecg)):  # && (to == 0)
        filter = np.dot([1, 4, 6, 4, 1, -1, -4, -6, -4, -1], ecg[(now-10):now])
        if (time % s2 == 0):
            # Adjust slope
            if (nslope == 0):
                slopecrit = max(slopecrit - slopecrit/16, scmin)
            elif (nslope >= 5):
                slopecrit = min(slopecrit + slopecrit/16, scmax)
        if (nslope == 0 and abs(filter) > slopecrit):
            nslope = nslope + 1
            maxtime = ms160
            if (filter > 0):
                sign = 1
            else:
                sign = -1
            qtime = time
        if (nslope != 0):
            if (filter * sign < -slopecrit):
                sign = -sign
                nslope = nslope + 1
                if (nslope > 4):
                    maxtime = ms200
                else:
                    maxtime = ms160
            elif (filter * sign > slopecrit and abs(filter) > maxslope):
                maxslope = abs(filter)
            if (maxtime < 0):
                if (2 <= nslope and nslope <= 4):
                    slopecrit = slopecrit + ((maxslope/4) - slopecrit)/8
                    if (slopecrit < scmin):
                        slopecrit = scmin
                    elif (slopecrit > scmax):
                        slopecrit = scmax
                    out.append(now - (time - qtime) - 4)
                    # annot.anntyp = NORMAL;
                    time = 0
                elif (nslope >= 5):
                    out.append(now - (time - qtime) - 4)
                    # annot.anntyp = ARFCT;
                nslope = 0
            maxtime = maxtime - 1
        time = time + 1
        now = now + 1

    # out = [x-1 for x in out] # adjust for 1 sample offset problem.

    if (debug is True):
        import matplotlib.pyplot as plt
        plt.plot(ecg, 'b', out, ecg[out], 'r*')
        plt.show()

    return out


VBL = 2048*2
blen = 250
def baseline(ecg, t):
    global blen
    global VBL
    if (baseline.t0 == 0):
        baseline.bbuf = [ecg[0] * (2*blen+1)]*VBL
    while (baseline.t0 < t and t+blen < len(ecg)):
        baseline.t0 += 1
        baseline.bbuf[baseline.t0 & (VBL-1)] += ecg[t+blen] - ecg[t-blen]
    return ((int)(baseline.bbuf[t & (VBL-1)]/blen))
baseline.t0 = 0
baseline.bbuf = []


def getx(ecg, t0, t1):
    x = 0.0
    for t in range(t0, t1+1):
        x += ecg[t] - baseline(ecg, t)
    return x


def edr(x):
    if (x == 0):
        return 0
    d = x - edr.xm
    if (edr.xc < 500):
        edr.xc += 1
        dn = d/edr.xc
    else:
        dn = d/edr.xc
        if (dn > edr.xdmax):
            dn = edr.xdmax
        elif (dn < -edr.xdmax):
            dn = -edr.xdmax
    edr.xm += dn
    edr.xd += abs(dn) - edr.xd/edr.xc
    if (edr.xd < 1.):
        edr.xd = 1.
    edr.xdmax = 3.*edr.xd/edr.xc
    r = d/edr.xd
    return ((int)(r*50.))
edr.xc = 0
edr.xd = edr.xdmax = edr.xm = 0.0

def group_epochs(signal, zeros, fs, epoch_length):
    samples_per_epoch = fs * epoch_length
    epoch_num = 1
    rin = []
    rout = []
    in_epoch = []
    out_epoch = []
    for i in range(0, len(zeros)-1):
        if median(signal[zeros[i]:zeros[i+1]]) >= 0:
            in_epoch.append(zeros[i+1] - zeros[i])
        else:
            out_epoch.append(zeros[i+1] - zeros[i])
        if zeros[i] > samples_per_epoch * epoch_num:
            epoch_num += 1
            rin.append(in_epoch)
            rout.append(out_epoch)
            in_epoch = []
            out_epoch = []
    if in_epoch or out_epoch:
        rin.append(in_epoch)
        rout.append(out_epoch)
    return rin, rout

def main(dataset, debug=False):
    ecg = dataset['ECG']

    # Detect onset of QRS complexes
    qrs = detectQRSPeaks(ecg, False)

    # EDR
    dt1 = -1
    dt2 = 1
    edrvals = []
    for sample in qrs:
        x = getx(ecg, sample+dt1, sample+dt2)
        edrvals.append(edr(x))

    # Spline interpolation
    breathing = interpolate.splrep(qrs, edrvals, s=0)
    s = range(0, len(ecg))
    interpolation = interpolate.splev(s, breathing, der=0)
    [_, filtered, zeros, _, _]= resp.resp(signal=interpolation, sampling_rate=250, show=False)

    if debug is True:
        # Plot result
        resp_ref = dataset['Resp']
        import matplotlib.pyplot as plt
        plt.plot(qrs, edrvals, 'g', label="raw edr")
        plt.plot(s, filtered, 'b', label="spline + filter")
        plt.plot(s, resp_ref, 'r', label="reference")
        plt.plot(zeros, resp_ref[zeros], 'm*', label="in - out")
        plt.legend()
        plt.show()

    return group_epochs(filtered, zeros, 250, 30)

if __name__ == "__main__":
    # Load datas
    mat_file = 'slpdb.mat'
    dataset = loadData(mat_file)
    main(dataset, debug=True)
