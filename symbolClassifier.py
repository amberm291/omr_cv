from pandas import read_csv
import csv
import os
import numpy as np
import cv2
import pandas as pd
import sklearn
from sklearn import svm
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
# import ipdb

class symbolClassifier:
	def __init__(self, model='rf', func='test'):
		self.model=model
		self.func=func
		self.validate=False
		self.clf=None
		self.cval_score = 0
		self.probabilities=[]
		if func=='validate':
			self.validate=True
		if model=='rf':
			if func=='test':
				filename = 'RF_trained_model.sav'
				self.clf = pickle.load(open(filename, 'rb'))
			else:
				self.train_RF('training.csv',self.validate)
				filename = 'RF_trained_model.sav'
				self.clf = pickle.load(open(filename, 'rb'))
		elif model=='svm':
			if func=='test':
				filename = 'SVM_trained_model.sav'
				self.clf = pickle.load(open(filename, 'rb'))
			else:
				self.train_SVC('training.csv',self.validate)
				filename = 'SVM_trained_model.sav'
				self.clf = pickle.load(open(filename, 'rb'))
		self.label_directory = read_csv('label_dict.csv')

	def extract_hog(self, im):
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

	def train_SVC(self, csv_file="training.csv", cross_val=False):
		training_data = read_csv(csv_file)
		data_x = training_data.drop(['labels'],axis=1)
		data_y = training_data['labels']
		svm_clf = svm.SVC(
			C=1.0,
			kernel='rbf',
			degree=3,
			gamma='auto',
			probability=True)
		if cross_val:
			print("Running stratified 5-fold cross validation...")
			self.cval_score = np.mean(cross_val_score(svm_clf,data_x,data_y, cv=StratifiedKFold(n_splits=5,shuffle=False), verbose=10))
		print("Training SVM on full training data ...")
		svm_clf.fit(data_x,data_y)
		filename = 'SVM_trained_model.sav'
		print("Storing trained model in SVM_trained_model.sav")
		pickle.dump(svm_clf, open(filename, 'wb'))

	def train_RF(self, csv_file="training.csv", cross_val=False):
		training_data = read_csv(csv_file)
		data_x = training_data.drop(['labels'],axis=1)
		data_y = training_data['labels']
		rf_clf = RandomForestClassifier(
			n_estimators=100,
			max_depth=10, 
			random_state=0)
		if cross_val:
			print("Running stratified 5-fold cross validation...")
			self.cval_score = np.mean(cross_val_score(rf_clf,data_x,data_y, cv=StratifiedKFold(n_splits=5,shuffle=False), verbose=10))
		print("Training Random Forest classifier on full training data ...")
		rf_clf.fit(data_x,data_y)
		filename = 'RF_trained_model.sav'
		print("Storing trained model in RF_trained_model.sav")
		pickle.dump(rf_clf, open(filename, 'wb'))

	def predict_class(self, im):
		if self.clf is not None:
			test_hog = self.extract_hog(im)
			data_x = test_hog.reshape(1,-1)
			predicted_label = self.clf.predict(data_x)
			self.probabilities = self.clf.predict_proba(data_x)
			max_prob = np.amax(self.probabilities)
			label_name = self.label_directory['name_labels'][predicted_label[0]]
			return predicted_label[0], label_name, max_prob
		else:
			print("Error : No pre-existing classifier found, use train='True' while calling class constructor to create a model")
			return None

