# -*- coding: utf-8 -*-
"""
Created on Mon Dec 04 22:22:39 2017

@author: OH YEA
"""
import os
import numpy as np
import cv2
import pandas as pd

import sklearn
from pandas import read_csv
import os
from sklearn import svm
import pickle
from PIL import Image
# import ipdb


# oldDir  = os.getcwd()

def extract_hog(im):
    img = np.array(Image.fromarray(im).resize((100, 100), Image.ANTIALIAS))
    theWinSize = (100,100)
    blockSize = (10,10)
    blockStride = (5,5)
    cellSize = (5,5)
    nBins = 9
    derivAperture = 1
    winSigma = 4
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nLevels = 25

    theHOG = cv2.HOGDescriptor(theWinSize, blockSize, blockStride, cellSize, nBins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nLevels)
    tempHOG = theHOG.compute(img)
    (nfeat,xx) = tempHOG.shape
    HOG_feat = np.reshape(tempHOG,[1,nfeat])[0]
    return HOG_feat
    # featureDF = pd.DataFrame(data=hogFeatures)
    # featureDF.to_csv('testing.csv')