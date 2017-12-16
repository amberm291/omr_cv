DESCRIPTION :
Two python scripts

1. extractHogResize.py : Resizes all the images in a given folder and extracts HOG features
	- extract_hog() : calls resize function to resize all images to 100 x 100 and extracts hog features which it stores in the file 'testing.csv'
		- Parameters : foldername
		- return : null
	- resize_img() : helper function for extract_hog()
2. symbolClassifier.py : Train and store svm classifier model. Use this model to test on given test data
	- train_SVC() : trains svm classifier on full training data and stores trained model to 'SVM_trained_model.sav'
		- Parameters : csv_training_file (default='training.csv'), cross_validation (boolean)
		- returns : null
	- test_SVC() : tests svm model stored in 'SVM_trained_model.sav' on given testing data (csv file)
		- Parameters : csv_testing_file (default=testing.csv)
		- returns : probabilities, predicted_labels

USE : 
Store the test images in a folder __foldername__

1. import extract_hog from extractHogResize
2. import train_SVC, test_SVC from symbolClassifier
4. 
3. extract_hog(__foldername__)
4. train_SVC('training.csv',False) # alternatively skip this step as the trained model is already generated 
5. probabilities, predicted_labels = test_SVC('testing.csv')