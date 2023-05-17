import os
import cv2

# Input folder path containing the images
input_folder = "dataset-3/5"

# Output folder path to save the rotated images
output_folder = "daataset-3/5-rotated"

# Check if the output folder exists, if not create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over the images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Update with the supported image extensions
        # Get the full path of the image file
        image_path = os.path.join(input_folder, filename)

        print(image_path)
        
        image = cv2.imread(image_path)

        # Rotate the image by 90 degrees clockwise
        rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(os.path.join(output_folder, filename+"-90.jpg"), rotated_90)

        # Rotate the image by 180 degrees
        rotated_180 = cv2.rotate(image, cv2.ROTATE_180)
        cv2.imwrite(os.path.join(output_folder, filename+"-180.jpg"), rotated_180)

        # Rotate the image by 270 degrees clockwise
        rotated_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(os.path.join(output_folder, filename+"-270.jpg"), rotated_270)