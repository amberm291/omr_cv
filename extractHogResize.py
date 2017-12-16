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

from sklearn.neighbors import KNeighborsClassifier
from pandas import read_csv
import os
from sklearn import svm
import pickle
from PIL import Image
# import ipdb


# oldDir  = os.getcwd()

def resize_img(folder):
    files = os.listdir(folder)
    for theFile in files:
        img = Image.open(folder +'/'+ theFile)
        img = img.resize((100, 100), Image.ANTIALIAS)
        img.save(folder+'/'+theFile)

def extract_hog(folder):
    resize_img(folder)
    
    hogFeatures = [[]]
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
    files = os.listdir(folder)
    for theFile in files:
        im = cv2.imread(folder +'/'+ theFile, 0)
        tempHOG = theHOG.compute(im)
        (nfeat,xx) = tempHOG.shape
        tempHOG = np.reshape(tempHOG,[1,nfeat])[0]
        hogFeatures.append(tempHOG)
    hogFeatures.remove([])
    hogFeatures = np.array(hogFeatures)
    featureDF = pd.DataFrame(data=hogFeatures)
    featureDF.to_csv('testing.csv')