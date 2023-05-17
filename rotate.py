import os
import cv2

# Input folder path containing the images
input_folder = "dataset-4/5"

# Output folder path to save the transformed images
output_folder = "dataset-4/5"

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
        cv2.imwrite(os.path.join(output_folder, filename + "-90.jpg"), rotated_90)

        # Rotate the image by 180 degrees
        rotated_180 = cv2.rotate(image, cv2.ROTATE_180)
        cv2.imwrite(os.path.join(output_folder, filename + "-180.jpg"), rotated_180)

        # Rotate the image by 270 degrees clockwise
        rotated_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(os.path.join(output_folder, filename + "-270.jpg"), rotated_270)

        # Flip the image horizontally
        flipped_horizontal = cv2.flip(image, 1)
        cv2.imwrite(os.path.join(output_folder, filename + "-flipped-horizontally.jpg"), flipped_horizontal)

        # Flip the image vertically
        flipped_vertical = cv2.flip(image, 0)
        cv2.imwrite(os.path.join(output_folder, filename + "-flipped-vertically.jpg"), flipped_vertical)

        # Scale the image by a factor of 0.5
        scaled_down = cv2.resize(image, None, fx=0.5, fy=0.5)
        cv2.imwrite(os.path.join(output_folder, filename + "-scaled-down.jpg"), scaled_down)

        # Scale the image by a factor of 1.5
        scaled_up = cv2.resize(image, None, fx=1.5, fy=1.5)
        cv2.imwrite(os.path.join(output_folder, filename + "-scaled-up.jpg"), scaled_up)