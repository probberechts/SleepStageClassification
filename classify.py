# -*- coding: utf-8 -*-
"""
Classify a dataset.

@author: Pieter
"""

import sys
import getopt
import numpy as np
from data import loadData, groupByEpoch
from models import createNormalizedDataframe, savePredictions
from featureConstruction import dictionaryInitialization, dico_construction
from sklearn.externals import joblib
import matplotlib.pyplot as plt


def classify(inputfile='slpdb.mat', epoch_length=30, make_plot=True, save=True, outputfile='classify.txt'):
    # Load datas
    dataset = loadData(inputfile)
    X = groupByEpoch(dataset['ECG'], 250, epoch_length)

    # Feature dictionary construction
    d_test = dictionaryInitialization()
    d_test = dico_construction(X, d_test)
    Xarr_norm = createNormalizedDataframe(d_test)

    # Predecition
    model = joblib.load('model.pkl')
    pred_res = list()
    for j in range(0, len(X)):
        l = str(model.predict(Xarr_norm[j])[0])
        pred_res.append(l)

    if make_plot:
        # Plot result
        plot(pred_res, dataset['labels'], epoch_length)

    if save:
        # Write prediction into text file
        savePredictions(pred_res, outputfile)


def plot(stages, reference, epoch_length=30):
    z = (len(stages)+1)*30
    plt.xlabel('time (s)')
    plt.xlim(0, z)

    time = np.arange(30, z, 30)
    plt.ylabel('Sleep Stage')
    plt.plot(time, stages, drawstyle='steps')
    plt.plot(time, reference, drawstyle='steps', color='r')

    plt.yticks(np.arange(-1, 9))

    plt.title('Hypnogram')

    plt.show()


def main(argv):
    inputfile = ''
    outputfile = ''
    store = False
    plot = False
    try:
        opts, args = getopt.getopt(argv, "hi:o:p", ["ifile=", "ofile=", "plot"])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -o <outputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile> -o <outputfile>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            store = True
            outputfile = arg
        elif opt in ("-p", "--plot"):
            plot = True
    classify(inputfile, 30, plot, store, outputfile)


if __name__ == "__main__":
    main(sys.argv[1:])
