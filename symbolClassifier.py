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
# import ipdb


def train_SVC(csv_file="training.csv", cross_val=False):
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
		cval_score = np.mean(cross_val_score(svm_clf,data_x,data_y, cv=StratifiedKFold(n_splits=5,shuffle=False), verbose=10))
		print(cval_score)
	else:
		print("Training SVM on full training data ...")
		svm_clf.fit(data_x,data_y)
		filename = 'SVM_trained_model.sav'
		print("Storing trained model in SVM_trained_model.sav")
		pickle.dump(svm_clf, open(filename, 'wb'))

def train_RF(csv_file="training.csv", cross_val=False):
	training_data = read_csv(csv_file)
	data_x = training_data.drop(['labels'],axis=1)
	data_y = training_data['labels']
	rf_clf = RandomForestClassifier(
		n_estimators=10,
		max_depth=5, 
		random_state=0)
	if cross_val:
		print("Running stratified 5-fold cross validation...")
		cval_score = np.mean(cross_val_score(rf_clf,data_x,data_y, cv=StratifiedKFold(n_splits=5,shuffle=False), verbose=10))
		print(cval_score)
	else:
		print("Training Random Forest classifier on full training data ...")
		rf_clf.fit(data_x,data_y)
		filename = 'RF_trained_model.sav'
		print("Storing trained model in RF_trained_model.sav")
		pickle.dump(rf_clf, open(filename, 'wb'))

def test_SVC(test_hog):
	print("Testing with SVM...")
	data_x = test_hog.reshape(1,-1)
	filename = 'SVM_trained_model.sav'
	loaded_model = pickle.load(open(filename, 'rb'))
	predicted_label = loaded_model.predict(data_x)
	probabilities = loaded_model.predict_proba(data_x)
	max_prob = np.amax(probabilities)
	return predicted_label, max_prob

def test_RF(test_hog):
	print("Testing with Random Forest...")
	data_x = test_hog.reshape(1,-1)
	filename = 'RF_trained_model.sav'
	loaded_model = pickle.load(open(filename, 'rb'))
	predicted_label = loaded_model.predict(data_x)
	probabilities = loaded_model.predict_proba(data_x)
	max_prob = np.amax(probabilities)
	return predicted_label, max_prob
