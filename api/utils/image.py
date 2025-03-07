import os
import cv2
import numpy as np
import base64
from PIL import Image
from pytesseract import pytesseract
from skimage.transform import radon
from numpy import mean, argmax

try:
    from parabolic import parabolic

    def argmax(x):
        return parabolic(x, np.argmax(x))[0]
except ImportError:
    from numpy import argmax


def convert_from_base64(base64_str):
    try:
        # Decode base64 to bytes
        image_data = base64.b64decode(base64_str)

        # Convert bytes to NumPy array
        nparr = np.frombuffer(image_data, np.uint8)

        # Decode image using OpenCV
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Decoded image is None. Possibly invalid base64 data.")

        return image
    except Exception as e:
        print(f"Error decoding base64 to image: {e}")
        return None


def convert_to_base64(image, format=".jpg"):
    try:
        # Encode image as bytes
        success, buffer = cv2.imencode(format, image)
        if not success:
            raise ValueError("Image encoding failed.")

        # Convert to base64 string
        base64_str = base64.b64encode(buffer).decode("utf-8")
        return base64_str
    except Exception as e:
        print(f"Error encoding image to base64: {e}")
        return None


def rms_flat(arr):
    """Return the root mean square of all the elements of *a*, flattened out."""
    return np.sqrt(np.mean(np.abs(arr) ** 2))


def get_rotate_angle(image):
    """Calculate the angle of rotation required to deskew an image."""
    try:
        # Ensure grayscale conversion is only done if needed
        if len(image.shape) == 2:  # Image is already grayscale
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_array = np.array(gray, dtype=np.uint8)

        # Demean the image
        image_array = image_array - mean(image_array)

        # Perform Radon transform
        sinogram = radon(image_array)

        # Find the rotation angle
        rotation_arr = np.array([rms_flat(line) for line in sinogram.T])
        rotation = argmax(rotation_arr)

        # Convert to int
        angle = int(90 - rotation) if rotation > 0 else 0  

        # print(f'Rotation: {angle:.2f} degrees')
        return angle
    except Exception as e:
        print(f"Error calculating rotation angle: {e}")
        # Default to 0 degrees if an error occurs
        return 0  


def rotate_image(image):
    """Rotate an image to correct its orientation."""
    try:
        # Get rotation angle
        angle = get_rotate_angle(image)

        # Get image dimensions
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # Compute the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Perform the rotation
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return rotated
    except Exception as e:
        print(f"Error rotating image: {e}")
        # Return original image if an error occurs
        return image  


def process_image(image):
    """Processes an image: converts to grayscale, rotates, binarizes, finds contours, and crops."""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Rotate image
        rotated = rotate_image(gray)

        # Convert to binary image
        _, binary = cv2.threshold(rotated, 180, 255, cv2.THRESH_BINARY)

        # Find contours in binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            max_contour = max(contours, key=cv2.contourArea)

            # Get bounding box
            x, y, w, h = cv2.boundingRect(max_contour)

            # Draw bounding box
            cv2.rectangle(rotated, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Crop image
            cropped = rotated[y:y + h, x:x + w]
        else:
            # If no contours, keep the rotated image
            cropped = rotated  

        return gray, rotated, binary, cropped

    except Exception as e:
        print(f"Error processing image: {e}")
        # Return None values if an error occurs
        return None, None, None  


def extract_information_from_image(image, config="--psm 6", lang="vie"):
    """Extracts text from an image using Tesseract OCR."""
    try:
        # Open image
        # image = Image.open(processed_image_path)

        # Perform OCR
        ocr_output = pytesseract.image_to_string(image, config=config, lang=lang)

        return ocr_output.strip()
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        # Return an empty string if an error occurs
        return ""  


def show_image(show_str, image):
    # Show the image
    cv2.imshow(show_str, image)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()