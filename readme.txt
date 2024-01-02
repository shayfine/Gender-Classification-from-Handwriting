homework 3 HanwrittenLetterRecognizer

The authors' contact information
shay finegold , 311165609
Dan monsonego , 313577595

Description
A program that get full path as argument to a directory with directories named by labels
the program will create a new directory with directories named by labels that contain the preprocessed images
the program will get hog feature and split the data to train, val, test
and calculate best k for knn alghoritm for train and val 
than it will claculate the knn alghoritm with the best k for train and test
create confusion matrix and save it in 'confusion_matrix.csv' file
claculate accuracies by class and save it and the best k in 'results.txt' file

Environment
visual studio 


How to Run Your Program
Instructions :
1. Enter to the required work environment
2. Open inside it the file attached to the work3.py
3. Run the file through the terminal and send the following argument:
	1. full path to a directory with directories labels that contain the source images
	
Example for running in pycharm:
python knn_classifier.py <img_dir_path>