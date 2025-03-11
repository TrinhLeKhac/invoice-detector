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
from skimage.filters import threshold_sauvola
import pywt

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


def adjust_brightness(image, method='clahe'):
    """
    Adjusts brightness using either CLAHE or histogram equalization.
    Additionally applies Adaptive Gamma Correction to enhance visibility.

    Args:
        image (numpy.ndarray): Input image.
        method (str): Brightness correction method ('clahe' or 'histogram').

    Returns:
        numpy.ndarray: Brightness-adjusted image.
    """

    # Convert the image to HSV color space to adjust brightness in the Value (V) channel.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Apply the chosen brightness enhancement method.
    if method == 'histogram':  
        v = cv2.equalizeHist(v)  # Standard histogram equalization.
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization) improves brightness locally,
    # preventing over-enhancement in certain regions.
    elif method == 'clahe':  
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v = clahe.apply(v)

    # Adaptive Gamma Correction adjusts brightness adaptively based on image intensity.
    # If the image is too dark (mean < 100), apply a gamma of 1.5 to brighten it.
    gamma = 1.5 if np.mean(v) < 100 else 1.0  
    v = np.power(v / 255.0, gamma) * 255.0  # Apply gamma correction.
    v = np.clip(v, 0, 255).astype(np.uint8)  # Ensure pixel values remain within [0, 255].

    # Merge the adjusted Value channel back into the HSV image and convert it back to BGR.
    adjusted_hsv = cv2.merge([h, s, v])
    adjusted_image = cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)

    return adjusted_image


def denoise_image(image, method="bilateral"):
    """
    Applies noise reduction on a grayscale image using an optimal denoising technique.

    Args:
        image (numpy.ndarray): Grayscale input image or BGR image.
        method (str): Denoising method: "bilateral", "nl_means", "wavelet".

    Returns:
        numpy.ndarray: Denoised image.
    """

    # Convert image to grayscale if it's not already
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == "bilateral":
        # Bilateral filter: Removes noise while preserving edges
        denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    elif method == "nl_means":
        # Non-Local Means Denoising: Effective for Gaussian noise
        denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    elif method == "wavelet":
        # Wavelet denoising: Advanced method for preserving fine details
        coeffs = pywt.wavedec2(gray, 'db1', level=2)
        coeffs_H = list(map(lambda x: tuple(np.where(np.abs(x) < 10, 0, x) for x in x), coeffs[1:]))
        denoised = pywt.waverec2((coeffs[0], *coeffs_H), 'db1')
        denoised = np.clip(denoised, 0, 255).astype(np.uint8)

    else:
        raise ValueError("Invalid method! Choose from 'bilateral', 'nl_means', or 'wavelet'.")

    return denoised


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


def enhance_and_binarize(image):
    """
    Enhances brightness using Adaptive Gamma Correction and applies adaptive binarization.
    
    Args:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Binary image after enhancement.
    """

    # Convert image to grayscale if it's not already
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ---- Step 1: Adaptive Gamma Correction ----
    mean_intensity = np.mean(gray)
    gamma = 1.5 if mean_intensity < 100 else 1.0  
    gamma_corrected = np.power(gray / 255.0, gamma) * 255.0
    gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)

    # ---- Step 2: Adaptive Thresholding ----
    binary_adaptive = cv2.adaptiveThreshold(
        gamma_corrected, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 31, 10  # Fine-tune blockSize & C
    )

    # # ---- Step 3: Otsu’s Thresholding (Optional) ----
    # _, binary_otsu = cv2.threshold(
    #     gamma_corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    # )

    # ---- Step 4: Sauvola Binarization (Advanced) ----
    window_size = 25  # Window size for local thresholding
    thresh_sauvola = threshold_sauvola(gamma_corrected, window_size=window_size)
    binary_sauvola = (gamma_corrected > thresh_sauvola).astype(np.uint8) * 255

    # Combine the best binarization results (choose based on testing)
    final_binary = cv2.bitwise_and(binary_adaptive, binary_sauvola)

    return final_binary

def preprocess_binary_image(binary_image):
    """
    1. Preprocess the binary image to remove small noise and enhance text contours. 
    2. Detects the most relevant contour that likely contains the text in the invoice.
    3. Merge overlapping contours into a single bounding box.
    4. Extracts the main text region from the binary image.

    Args:
        binary_image (numpy.ndarray): Input binary image (0 and 255 values).
    
    Returns:
        numpy.ndarray: Processed binary image.
    """
    # Remove small noise using morphological opening
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)

    # Dilate to connect broken characters
    cleaned = cv2.dilate(cleaned, kernel, iterations=1) # (1)

    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get image dimensions
    height, width = cleaned.shape

    # Filter contours based on area and aspect ratio
    potential_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / h if h > 0 else 0
        
        # Heuristic filtering (remove very small or large regions)
        if area > 500 and area < (0.9 * height * width) and 0.2 < aspect_ratio < 10:
            potential_contours.append((x, y, w, h))

    if not potential_contours:   # (2)
        return None  # No valid contours found

    # Merge overlapping contours
    def merge_contours(contours):
    
        x_min = min([c[0] for c in contours])
        y_min = min([c[1] for c in contours])
        x_max = max([c[0] + c[2] for c in contours])
        y_max = max([c[1] + c[3] for c in contours])

        return (x_min, y_min, x_max - x_min, y_max - y_min)

    # Merge overlapping contours into a single bounding box.
    text_contour = merge_contours(potential_contours)  # (3)

    if text_contour: # (4)
        x, y, w, h = text_contour
        cropped_binary_image = binary_image[y:y+h, x:x+w]
        return text_contour, cropped_binary_image
    else:
        return None, cropped_binary_image


def process_image(image, denoise=True, binary_enhance=True, rec_ctour=True, is_traditional=False):
    """Processes an image: converts to grayscale, rotates, binarizes, finds contours, and crops."""
    try:
        # Adjust brightness
        adjusted_image = adjust_brightness(image, method='clahe')

        # Convert to grayscale
        gray = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)

        # Denoise image
        if denoise:
            denoised = denoise_image(gray, method="nl_means") # bilateral ~ nl_means > wavelet
        else:
            denoised = gray

        # Rotate image
        rotated = deskew_image(denoised)

        # Convert to binary image
        if binary_enhance:
            # Binary: Làm rõ mặt chữ, các vùng xung quanh giữ lại đường net => contour không focus vào information (nếu dùng cv2.coutourArea)
            binary = enhance_and_binarize(rotated)
        else:
            # Binary: Vùng xung quanh (thùng hàng, ..) bị lem => contours chính xác => crop OCR focus được vào information
            # Tuy có lúc bị lem nhiều => cắt quá đà, bị mất thông tin
            _, binary = cv2.threshold(rotated, 180, 255, cv2.THRESH_BINARY) 
        
        if is_traditional: # Use style traditional
            # Find contours in binary image
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:

                # rectangle coutour
                if rec_ctour: 
                    max_contour = max(contours, key=cv2.contourArea)

                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(max_contour)

                    # Draw bounding box
                    cv2.rectangle(rotated, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    # Crop image
                    cropped = rotated[y:y + h, x:x + w]

                # poly contour
                else:
                    # Gộp tất cả contour thành 1 vùng lớn
                    all_contours = np.vstack(contours)

                    # Tạo đa giác lồi (convex hull) để bao quanh vùng chữ
                    hull = cv2.convexHull(all_contours)

                    # Tạo mask với vùng đa giác
                    mask = np.zeros_like(binary)
                    cv2.fillPoly(mask, [hull], 255)

                    # Cắt vùng ảnh theo mask
                    cropped = cv2.bitwise_and(rotated, rotated, mask=mask)
            else:
                # If no contours, keep the rotated image
                cropped = rotated  

        else: # new style
            text_contour, cropped_binary_image = preprocess_binary_image(binary)

            if text_contour is not None:
                x, y, w, h = text_contour
                cropped = rotated[y:y + h, x:x + w]
                binary = cropped_binary_image
            else:
                # If no contours, keep the rotated image
                cropped = rotated

        return denoised, binary, cropped # rotated

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