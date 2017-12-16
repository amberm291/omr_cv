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
		C=1.0
		,kernel='rbf'
		,degree=3
		,gamma=auto
		,probability=True)
	if cross_val:
		print("Running stratified 5-fold cross validation...")
		cval_score = np.mean(cross_val_score(svm_clf,data_x,data_y, cv=StratifiedKFold(n_splits=5,shuffle=False), verbose=10))
		print(cval_score)
	else
		print("Training SVM on full training data ...")
		svm_clf.fit(data_x,data_y)
	filename = 'SVM_trained_model.sav'
	print("Storing trained model in SVM_trained_model.sav")
	pickle.dump(svm_clf, open(filename, 'wb'))

def test_SVC(test_csv="testing.csv"):
	print("Testing with SVM...")
	outputDF = pd.DataFrame()
	# outputdf = pd.DataFrame(data=test_x)
	data_x = read_csv(test_csv)
	filename = 'SVM_trained_model.sav'
	loaded_model = pickle.load(open(filename, 'rb'))
	predicted_labels = loaded_model.predict(data_x)
	outputdf['predicted_labels'] = predicted_labels
	result = loaded_model.predict_proba(data_x)
	probabilityDF = pd.DataFrame(data=result)
	print("Writing predictions to predictions.csv...")
	outputdf.to_csv('predictions.csv')
	print("Returning (probabilities,labels)...")
	return result, predicted_labels
