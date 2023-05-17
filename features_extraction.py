import cv2
from skimage.feature import hog
from sklearn.decomposition import PCA
import numpy as np



def feature_extraction(img):
    # set the parameters for HoG
    orientations = 9
    pixels_per_cell = (50, 50)
    cells_per_block = (3, 3)
    visualize = False
    transform_sqrt = False

    # calculate the HoG features for the image
    hog_result = hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=visualize,
        transform_sqrt=transform_sqrt,
        block_norm='L2-Hys',
        feature_vector=True
    )


    features = hog_result.ravel()

    return features

def apply_PCA(features, pca):
    # apply PCA
    test_X = features.reshape(1, -1)
    reducded_features = pca.transform(test_X)
                                   
    return reducded_features

