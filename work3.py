import pandas as pd
import numpy as np
import os
import cv2
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import confusion_matrix
from skimage.io import imread
from skimage.feature import hog


def getHogFeature(output_folder_path):
    fe_lst, la_lst = [], []
    for label_ in os.listdir(output_folder_path):
        for img in os.listdir(output_folder_path + label_):
            imgTemp = imread(output_folder_path + label_ + '/' + img)
            fd, hog_image = hog(imgTemp, orientations=9, pixels_per_cell=(3, 3),
                                cells_per_block=(1, 1), visualize=True)
            fe_lst.append(fd)
            la_lst.append(label_)
    X = pd.DataFrame(fe_lst, columns=[str(x)
                     for x in list(range(0, len(fe_lst[0])))])
    y = (pd.DataFrame(la_lst, columns=['label']))['label']
    return X, y


def padding_image(img):  # padding the image
    shape = img.shape
    if shape[0] == shape[1]:
        return img
    elif shape[0] > shape[1]:
        distance = shape[0] - shape[1]
        top = bottom = 0
        left = distance // 2
        right = distance - left
    else:
        distance = shape[1] - shape[0]
        left = right = 0
        top = distance // 2
        bottom = distance - top
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])


def calculate_best_k(X_train, y_train, X_test, y_test):
    liy = []
    acc, k_max_acc = 0, 1
    for k in range(1, 16, 2):
        knn = KNN(n_neighbors=k, metric='euclidean')
        knn.fit(X_train, y_train)
        liy.append(knn.score(X_test, y_test))
        if liy[-1] > acc:
            acc = liy[-1]
            k_max_acc = k
    return k_max_acc


def main(input_folder):
    # new folder for the processed images
    output_folder = './hhd_dataset_processed/'
    os.mkdir(output_folder)  # making the folder
    for i in range(27):
        os.mkdir(output_folder + '/' + str(i))
    print('Output folder has been created')

    for label in os.listdir(input_folder):   # starting the pre-processing step
        for img in os.listdir(input_folder + label):
            imgTemp = cv2.imread(input_folder + label + '/' + img)
            # converting to gray
            imgGray = cv2.cvtColor(imgTemp, cv2.COLOR_BGR2GRAY)
            imgBlur = cv2.blur(imgGray, (10, 10))  # bluring the image
            thresh, imgB = cv2.threshold(
                imgBlur, 120, 255, cv2.THRESH_OTSU)  # threshold
            imgPadding = padding_image(imgB)  # padding the image
            imgResized = cv2.resize(imgPadding, (32, 32))  # resazing the image
            cv2.imwrite(output_folder + label + '/' + img, imgResized)

    X, y = getHogFeature(output_folder)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=9177391, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=9177391, stratify=y_test)
    m = calculate_best_k(X_train, y_train, X_val, y_val)
    knn = KNN(m, metric='euclidean')
    knn.fit(X_train, y_train)
    knn.score(X_test, y_test)
    y_pred = knn.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(cm).to_csv('confusion_matrix.csv')
    per_class_accuracies = {}
    for idx, cls in enumerate(set(y)):
        true_negatives = np.sum(
            np.delete(np.delete(cm, idx, axis=0), idx, axis=1))
        true_positives = cm[idx, idx]
        per_class_accuracies[cls] = (
            true_positives + true_negatives) / np.sum(cm)
    sortedlist = [(k, per_class_accuracies[k])
                  for k in sorted(per_class_accuracies, key=int)]
    output = open("results.txt", "w")
    output.writelines(f'k = {m}\n')
    output.writelines('Letter\t\t\tAccuracy\n')
    for k, v in dict(sortedlist).items():
        output.writelines(f'{k}\t\t\t{v}\n')


if __name__ == '__main__':
    main(str((sys.argv[1])+'/'))
