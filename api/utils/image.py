import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from PIL import Image
from pytesseract import pytesseract
from numpy import mean, argmax


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


def get_rotation_angle(image, angle_threshold=15):
    """
    Find the skew angle of the image using Hough Line Transform.
    This method detects straight lines and estimates the median rotation angle.
    """
    # Convert image to grayscale if it's not already
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    angles = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            angle = np.rad2deg(theta) - 90  # Convert radian to degree

            # Filter out extreme angles that might cause incorrect rotations
            if -angle_threshold < angle < angle_threshold:  
                angles.append(angle)

    if angles:
        median_angle = np.median(angles)
        
        # Ensure that the angle is not leading to incorrect horizontal rotations
        if (median_angle < -angle_threshold) or (median_angle > angle_threshold):
            return 0  # Ignore extreme incorrect angles
        
        return median_angle  

    return 0  # Return 0 if no rotation is detected


def rotate_image(image, angle):
    """
    Rotate the image by the given angle while keeping the entire content visible.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute the new bounding dimensions
    cos_theta = np.abs(rotation_matrix[0, 0])
    sin_theta = np.abs(rotation_matrix[0, 1])

    new_w = int((h * sin_theta) + (w * cos_theta))
    new_h = int((h * cos_theta) + (w * sin_theta))

    # Adjust rotation matrix to take into account translation
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated_image


def deskew_image(image):
    """
    Detect and correct the skew of an image.
    Ensures that the rotation does not flip the image sideways.
    """
    angle = get_rotation_angle(image)  # Detect rotation angle
    print(angle)
    if abs(angle) < 0.1:  # If angle is too small, no rotation is needed
        return image

    corrected_image = rotate_image(image, angle)
    return corrected_image



def process_image(image):
    """Processes an image: converts to grayscale, rotates, binarizes, finds contours, and crops."""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Rotate image
        rotated = deskew_image(gray)

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

        return gray, rotated, cropped # binary

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