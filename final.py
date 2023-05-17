import os
import pickle
import time
from Preprocess import *
from features_extraction import *


# Load the trained model from the pickle file
model = pickle.load(open('SVC.sav', 'rb'))
# Load the PCA model
pca = pickle.load(open('pca_model.pkl', 'rb'))

# Define the directory containing the test images
test_dir = 'data'

# Create a list of test image file names sorted in increasing order
test_files = sorted(os.listdir(test_dir), key=lambda x: int(os.path.splitext(x)[0]))

# Create a file to save the prediction results
results_file = open('results.txt', 'w')

# Create a file to save the time taken for each iteration
time_file = open('time.txt', 'w')

# Define the image extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

# Loop over all test images in increasing order
for filename in test_files:
    if any(filename.lower().endswith(extension) for extension in image_extensions):
        # Read the test image
        image_path = os.path.join(test_dir, filename)

        # Start time
        start_time = time.time()

        # Call preprocess function on the image (replace with your preprocess function)
        preprocessed_image = preprocess(image_path)

        # Call feature extraction function on the preprocessed image (replace with your feature extraction function)
        features = feature_extraction(preprocessed_image)

        # Call PCA on the features (replace with your PCA function)
        features = apply_PCA(features.reshape(1, -1), pca)

        # Use the loaded model to predict the class
        prediction = model.predict(features)

        # Save the prediction in results.txt
        results_file.write(str(prediction[0]) + '\n')
        # End time
        end_time = time.time()

        # Calculate the time taken for this iteration and save it in time.txt
        iteration_time = round(end_time - start_time, 3)
        time_file.write(str(iteration_time) + '\n')

# Close the result and time files
results_file.close()
time_file.close()
