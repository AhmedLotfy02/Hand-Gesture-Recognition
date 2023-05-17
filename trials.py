
import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage import io
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import time
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

################OLD PREPROCESSING#####################
# define the target size of the preprocessed images
target_size = (64, 64)

# define the Hand Segmentation function
def handSegmentation(image):

    image[(image[:, :, 0] < 80) & (image[:, :, 1] < 80) & (image[:, :, 2] < 80)] = (200, 200, 200)
    
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a binary mask of pixels within the skin color range
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply morphological operations to reduce noise and improve the mask

    # Find contours in the skin mask
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area (assumed to be the hand)
    hand_contour = max(contours, key=cv2.contourArea)

    # Create a blank mask for the hand region
    hand_mask = np.zeros_like(image, dtype=np.uint8)

    # Draw the hand contour on the mask
    cv2.drawContours(hand_mask, [hand_contour], 0, (0, 255, 0), thickness= 2)

    # Apply the mask to the original image to extract the hand region
    hand_segmented = cv2.bitwise_and(image, hand_mask)
    
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (6,6))
    skin_mask = cv2.morphologyEx(hand_segmented, cv2.MORPH_ERODE, kernel)
    skin_mask = cv2.morphologyEx(hand_segmented, cv2.MORPH_DILATE, kernel2)
    
    return skin_mask

# define the preprocessing functions
def convert_to_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray)
    return gray


def apply_gaussian_blur(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    return blurred


def apply_threshold(img):
    _, thresh = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return thresh


def find_contours(img):
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy


def find_largest_contour(contours):
    max_contour = max(contours, key=cv2.contourArea)
    return max_contour


def create_mask(img, contour):
    mask = cv2.drawContours(np.zeros_like(
        img), [contour], 0, (255, 255, 255), -1)
    return mask


def apply_mask(img, mask):
    masked_image = cv2.bitwise_and(img, img, mask=mask)
    return masked_image


def resize_image(img, size):
    resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return resized


def normalize_image(img):
    normalized = img.astype("float32") / 255.0
    return normalized



# Old preprocessing phase
def preprocess_old(img_path):
    img = io.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = apply_gaussian_blur(gray)
    # blurred = np.uint8(blurred)
    thresh = apply_threshold(blurred)
    # contours, hierarchy = find_contours(thresh)
    # max_contour = find_largest_contour(contours)
    # mask = create_mask(gray, thresh)
    # masked_image = apply_mask(img, mask)
    resized = resize_image(thresh, target_size)
    normalized = normalize_image(resized)

    return normalized


def preprocess_old_2(img_path):
    img = cv2.imread(img_path)


    # 1 Image Rescaling
    img = cv2.resize(img, (320, 200))


    img_gray = handSegmentation(img)

    # # 2 Image Enhancement
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

    img_gray[img_gray < 50] = 0
    # gamma = 2

    # # # Calculate the lookup table
    # lookup_table = np.zeros((256, 1), dtype='uint8')
    # for i in range(256):
    #     lookup_table[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)

    # # Apply the lookup table
    # img_gray = cv2.LUT(img_gray, lookup_table)

    # img_gray = cv2.equalizeHist(img_gray)

    # # 3 Background Subtraction
    # fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=30)
    # fgmask = fgbg.apply(img_gray)
    # img_gray = cv2.bitwise_and(img_gray, img_gray, mask=fgmask)

    # # #4 Noise Reduction
    # # Blur
    img_gray = cv2.medianBlur(img_gray, 3)

    return img_gray



# old Feature Extraction Functions
def get_hog_features(img):
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size,
                            block_stride, cell_size, nbins)
    hog_feats = hog.compute(img)
    return hog_feats


def get_sift_features(img):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return descriptors


# set the parameters for SIFT
nfeatures = 200
nOctaveLayers = 3
contrastThreshold = 0.04
edgeThreshold = 10
sigma = 1.6

sift = cv2.SIFT_create(
    nfeatures=nfeatures,
    nOctaveLayers=nOctaveLayers,
    contrastThreshold=contrastThreshold,
    edgeThreshold=edgeThreshold,
    sigma=sigma
)



# Classification Functions
def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
    return X_train, X_test, y_train, y_test

def split_validation_test(X, y):
    X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=0.5, random_state=21)
    return X_val, X_test, y_val, y_test

def svm_model(X_train, y_train, X_test):
    clf = svm.SVC(kernel='rbf', C=100, gamma=0.5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    y_pred_train = clf.predict(X_train)

    # filename = 'SVC.sav'
    # pickle.dump(clf, open(filename, 'wb'))

    return y_pred, y_pred_train


def calc_accuracy(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


## Some Models that didn't work well

# Accuracy was 70%
from sklearn.linear_model import SGDClassifier
def adaboost(X_train, y_train, X_test):

    clf = AdaBoostClassifier(SGDClassifier(loss='hinge'), algorithm='SAMME')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

# Accuracy was 63%
def knn_model(X_train, y_train, X_test):
    # Create the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=11)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = knn.predict(X_test)

    return y_pred

# Accuracy was 68%
def random_forest_model(X_train, y_train, X_test):
    rfc = RandomForestClassifier(max_depth=None, max_features='sqrt',
                                 min_samples_leaf=2, min_samples_split=2, n_estimators=200)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)

    # filename = 'RFC.sav'
    # pickle.dump(clf, open(filename, 'wb'))

    return y_pred



# Parmaters Tuning
# Grid Search
C_range = [0.01, 0.1, 1, 10, 100, 1000]
gamma_range = [1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.5]
parameters = [ # gamma = 0.5, C = 100, rbf
    # 'C': 10, 'gamma': 0.5, 'kernel': 'poly'
    {
        'kernel': ['rbf'],
        'gamma': [1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.5],
        'C': [0.01, 0.1, 1, 10, 100, 1000]
    },
    #  {
    #     'kernel': ['linear'],
    #     'C': [0.01, 0.1, 1, 10, 100, 1000, 10000]
    #  },
    #  {
    #     'kernel': ['sigmoid'],
    #     'gamma': [1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.5],
    #     'C': [0.01, 0.1, 1, 10, 100, 1000, 10000]
    #  },
   #   {
   #      'kernel': ['poly'],
   #      'gamma': [1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.5],
   #      'C': [0.01, 0.1, 1, 10, 100, 1000, 10000]
   #   }

]