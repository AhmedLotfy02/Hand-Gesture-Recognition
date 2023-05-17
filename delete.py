import os

# Folder path containing the images
folder_path = "dataset-4/2"

# Iterate over the images in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Update with the supported image extensions
        # Check if the filename contains "-180", "-90", or "-270"
        if "-180" in filename or "-90" in filename or "-270" in filename:
            # Get the full path of the image file
            image_path = os.path.join(folder_path, filename)

            # Delete the image file
            os.remove(image_path)

            print(f"Deleted: {filename}")