import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
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


def convert_from_base64(full_base64_string):
    try:
        match = re.match(r"data:image/(?P<format>png|jpeg|jpg);base64,(?P<data>.+)", full_base64_string)
    
        if not match:
            raise ValueError("Invalid base64 string or wrong format image!")

        # Get format (png, jpeg, jpg)
        image_format = match.group("format")

        # Get data
        base64_string = match.group("data")

        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_string)

        # Convert bytes to NumPy array
        image_nparr = np.frombuffer(image_bytes, np.uint8)

        # Decode image using OpenCV
        image = cv2.imdecode(image_nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Decoded image is None.")

        return image
    except Exception as e:
        print(f"Error decoding base64 to image: {e}")
        return None


def convert_to_base64(image, format=".jpg"):
    try:
        # Encode image as bytes
        success, encoded_image = cv2.imencode(format, image)
        if not success:
            raise ValueError("Image encoding failed.")

        # Convert to base64 string
        base64_str = base64.b64encode(encoded_image).decode("utf-8")

        # Determine MIME type based on format
        format = format.lower().replace(".", "")  # Remove leading dot (".jpg" → "jpg")
        if format == "jpg":
            format = "jpeg"  # Convert "jpg" to "jpeg" for correct MIME type

        # Add prefix for Data URI
        return f"data:image/{format};base64,{base64_str}"
    
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
        # rotated = rotate_image(gray)

        # Convert to binary image
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY) # rotated

        # Find contours in binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            max_contour = max(contours, key=cv2.contourArea)

            # Get bounding box
            x, y, w, h = cv2.boundingRect(max_contour)

            # Draw bounding box
            cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 255), 2) # rotated

            # Crop image
            cropped = gray[y:y + h, x:x + w] # rotated
        else:
            # If no contours, keep the rotated image
            cropped = gray  

        return gray, binary, cropped # rotated

    except Exception as e:
        print(f"Error processing image: {e}")
        # Return None values if an error occurs
        return None, None, None, None  


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
    try:
        # Show the image
        cv2.imshow(show_str, image)

        # Wait for a key press and close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error processing image: {e}")
        # If you are running the script on a server 
        # without a graphical interface (e.g., Linux server, WSL, Docker),
        # cv2.imshow() will fail

        # Convert from BGR to RGB
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  
        plt.title(show_str)
        plt.axis("off")
        plt.show()