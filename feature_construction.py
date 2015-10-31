# -*- coding: utf-8 -*-
"""
Construct a dictionary with all features.

@author: Pieter
"""

from ECGutils import ECGFeatures


def dictionaryInitialization():
    d = {
        'mean': list(),
        'std': list(),
        'CV': list(),
        'LF': list(),
        'HF': list(),
        'FreqmaxP': list(),
        'maxHFD': list(),
        'LFHFratio': list(),
        'inter': list(),
        'H': list(),
        'pval951': list(),
        'pval952': list(),
        'dfaslope': list(),
        'dfaintercept': list(),
        'dfaslopeT': list(),
        'dfainterceptT': list(),
        'alpha1': list(),
        'alpha2': list(),
        'alpha3': list(),
        'P11': list(),
        'P12': list(),
        'P13': list(),
        'freq1': list(),
        'freq2': list(),
        'freq3': list(),
    }
    return d


def dico_construction(X, dd):
    d = dd
    for i in range(0, len(X)):
        fea = ECGFeatures(X[i], 250)
        d['mean'].append(fea.mean)
        d['std'].append(fea.std)
        d['CV'].append(fea.CV)
        d['LF'].append(fea.LF)
        d['HF'].append(fea.HF)
        d['FreqmaxP'].append(fea.FreqmaxP)
        d['maxHFD'].append(fea.maxHFD)
        d['LFHFratio'].append(fea.LFHFratio)
        d['inter'].append(fea.inter)
        d['H'].append(fea.H)
        d['pval951'].append(fea.pval951)
        d['pval952'].append(fea.pval952)
        d['dfaslope'].append(fea.dfaslope)
        d['dfaintercept'].append(fea.dfaintercept)
        d['dfaslopeT'].append(fea.dfaslopeT)
        d['dfainterceptT'].append(fea.dfainterceptT)
        d['alpha1'].append(fea.alpha1)
        d['alpha2'].append(fea.alpha2)
        d['alpha3'].append(fea.alpha3)
        d['P11'].append(fea.P11)
        d['P12'].append(fea.P12)
        d['P13'].append(fea.P13)
        d['freq1'].append(fea.freq1)
        d['freq2'].append(fea.freq2)
        d['freq3'].append(fea.freq3)
    return d

'''
do = dictionaryInitialization()
do = dico_train_construction(dataset['X_train'],do)

do_test = dictionaryInitialization()
do_test = dico_train_construction(dataset['X_test'],do_test)
'''
