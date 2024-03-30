from PIL import Image
import os
import shutil

def del_non_square_images(input_folder):
    for file in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file)
        try:
            with Image.open(file_path) as img:
                # Check if the image is square
                if img.width != img.height:
                    # Prepare the target path
                    print("non-square image found")                    
            # DEL operation outside the 'with' block
            if img.width != img.height:
                os.remove(file_path)
                print(f"Deleted non-square image: {file}")
        except Exception as e:
            print(f"Error processing file {file}: {e}")

input_folder = "COCO_OUT"  # Your input folder
del_non_square_images(input_folder)
