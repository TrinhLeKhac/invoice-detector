import os
import cv2
from utils.image.helper import detect_and_crop_invoice


def crop_all_images(input_dir, output_dir):

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get list of all image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_name in image_files:
        image_path = os.path.join(input_dir, image_name)
        output_path = os.path.join(output_dir, image_name)

        # Read the image
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error reading image: {image_path}")
            continue

        # Extract crop region
        cropped_image = detect_and_crop_invoice(image)

        try:
            # Save the cropped image
            cv2.imwrite(output_path, cropped_image)
            print(f"Saved cropped image: {output_path}")
        except:
            print("Something wrong")

    
if __name__ == "__main__":
    input_dir = "test/images"
    output_dir = "test/cropped"
    crop_all_images(input_dir, output_dir)