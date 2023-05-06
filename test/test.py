import os
import cv2
from skimage import feature

import numpy as np

# define the path to the input images
input_path = r"E:\CMP3\Second term\Neural Networks\Project\Project_Dataset_S23-20230429T055712Z-002\Project_Dataset_S23\dataset_sample\men\input"

# define the path to the output preprocessed images
output_path = r"E:\CMP3\Second term\Neural Networks\Project\Project_Dataset_S23-20230429T055712Z-002\Project_Dataset_S23\dataset_sample\men\newoutput"
# define the target size of the preprocessed images
target_size = (64, 64)

# define the preprocessing functions


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


print(cv2.__version__)
# loop over the input images
for filename in os.listdir(input_path):
    # load the image
    img_path = os.path.join(input_path, filename)
    # print(img_path)

    img = cv2.imread(img_path)

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
    hog_features = feature.hog(gray2, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), block_norm='L2-Hys',
                               visualize=False, transform_sqrt=True)

    # Initialize a SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute SIFT descriptors
    keypoints, sift_features = sift.detectAndCompute(gray2, None)

    # Concatenate HoG and SIFT features
    features = np.concatenate((hog_features, sift_features), axis=0)

    # Print the shape of the concatenated feature vector
    print('Shape of concatenated feature vector:', features.shape)

# save the preprocessed image
# output_filename = os.path.join(output_path, filename)
# cv2.imwrite(output_filename, normalized * 255)

# import cv2
# import numpy as np
#
# # Load input image
# img = cv2.imread(
#     r"E:\CMP3\Second term\Neural Networks\Project\Project_Dataset_S23-20230429T055712Z-002\Project_Dataset_S23\dataset_sample\men\input\0_men (2).jpg")
#
#
# # Convert BGR image to HSV color space
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
# # Define lower and upper skin color ranges in HSV
# lower_skin = np.array([0, 20, 70], dtype=np.uint8)
# upper_skin = np.array([20, 255, 255], dtype=np.uint8)
#
# # Create a binary mask of the skin color region in the image
# mask = cv2.inRange(hsv, lower_skin, upper_skin)
#
# # Find the contours of the largest connected component in the mask
# contours, hierarchy = cv2.findContours(
#     mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# max_contour = max(contours, key=cv2.contourArea)
#
# # Create a binary image of the hand region
# hand_segment = np.zeros_like(img)
# cv2.drawContours(hand_segment, [max_contour], 0, (255, 255, 255), cv2.FILLED)
# hand_segment = cv2.cvtColor(hand_segment, cv2.COLOR_BGR2GRAY)
# hand_segment = cv2.threshold(
#     hand_segment, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#
# # Apply distance transform to the binary hand image
# dist_transform = cv2.distanceTransform(hand_segment, cv2.DIST_L2, 5)
#
# # Threshold the distance transform image to obtain a segmentation of the hand
# seg_threshold = 0.2 * dist_transform.max()
# seg_mask = np.uint8(dist_transform > seg_threshold)
# seg_mask = cv2.dilate(seg_mask, np.ones((3, 3), dtype=np.uint8), iterations=2)
# seg_mask = cv2.erode(seg_mask, np.ones((3, 3), dtype=np.uint8), iterations=2)
#
# # Convert the segmented hand image to BGR color space
# hand_segment = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2BGR)
#
# # Blend the segmented hand image with the original input image
# output_img = cv2.addWeighted(img, 0.7, hand_segment, 0.3, 0)
#
# # Save output image
# cv2.imwrite('output1.jpg', output_img)
