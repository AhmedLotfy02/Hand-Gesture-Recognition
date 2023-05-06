import cv2
import numpy as np
from sklearn import svm

# Define the list of images to process
images = ['./0_men (1).jpg', './0_men (2).jpg']

# define the target size of the preprocessed images
target_size = (64, 64)


def convert_to_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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


# Initialize the HoG and SIFT feature extractors
win_size = (64, 128)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
nbins = 9
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
sift = cv2.SIFT_create()

# Extract features for each image
all_feats = []
for image_path in images:
    # Load the image
    img = cv2.imread(image_path)
    # apply the preprocessing pipeline
    gray = convert_to_gray(img)
    blurred = apply_gaussian_blur(gray)
    thresh = apply_threshold(blurred)
    contours, hierarchy = find_contours(thresh)
    max_contour = find_largest_contour(contours)
    mask = create_mask(gray, max_contour)
    masked_image = apply_mask(img, mask)
    resized = resize_image(masked_image, target_size)
    normalized = normalize_image(resized)

    gray2 = convert_to_gray(normalized)
    # Extract the HoG features
    hog_feats = hog.compute(gray2)

    # Extract the SIFT features
    kp, sift_feats = sift.detectAndCompute(gray2, None)

    # Concatenate the features
    feats = np.concatenate((hog_feats.ravel(), sift_feats.ravel()))
    all_feats.append(feats)

# Create the SVM model and train it on the features
X = np.array(all_feats)
# Labels for the images (1 for positive, 0 for negative)
y = np.array([1, 0, 0])
svm_model = svm.SVC(kernel='linear', C=1)
svm_model.fit(X, y)

# Use the SVM model to classify new images
test_img = cv2.imread('test_image.jpg')
hog_feats_test = hog.compute(test_img)
kp_test, sift_feats_test = sift.detectAndCompute(test_img, None)
feats_test = np.concatenate((hog_feats_test.ravel(), sift_feats_test.ravel()))
X_test = feats_test.reshape(1, -1)
y_pred = svm_model.predict(X_test)
