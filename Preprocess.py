import cv2
import numpy as np
from skimage.feature import hog
from skimage import io
import numpy as np






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



def preprocess(img_path):
    img = cv2.imread(img_path)


    # 1 Image Rescaling
    img = cv2.resize(img, (320, 200))


    img_gray = handSegmentation(img)

    # 2 Image Enhancement
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

    img_gray[img_gray < 50] = 0

    # 3 Noise Reduction
    # Blur
    img_gray = cv2.medianBlur(img_gray, 3)

    return img_gray




