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


def detect_table(image):
    # Convert image to grayscale if it's not already
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # Làm mờ để giảm nhiễu
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Tìm contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sắp xếp contours theo diện tích (lớn nhất thường là hóa đơn)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Lấy contour có dạng hình chữ nhật (hóa đơn)
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:  # Nếu contour có 4 đỉnh, có thể là hóa đơn
            receipt_contour = approx
            break

    # Áp dụng Perspective Transform nếu tìm thấy hóa đơn
    if 'receipt_contour' in locals():
        pts = receipt_contour.reshape(4, 2)
        
        # Sắp xếp điểm theo thứ tự: [Top-left, Top-right, Bottom-right, Bottom-left]
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # Xác định kích thước hóa đơn mới
        width = 500
        height = 700
        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

        # Tính ma trận biến đổi và warp
        matrix = cv2.getPerspectiveTransform(rect, dst)
        cropped_receipt = cv2.warpPerspective(image, matrix, (width, height))

        return cropped_receipt


def process_image(image, denoise=True, binary_enhance=False, thickness=15):
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

        # Convert to binary image to FIND CONTOUR
        # Binary (1): Vùng xung quanh (thùng hàng, ..) bị lem 
        # => contours chính xác => crop OCR focus được vào information
        # Tuy nhiên có lúc bị lem nhiều => cắt quá đà, bị mất thông tin
        _, binary = cv2.threshold(rotated, 180, 255, cv2.THRESH_BINARY) 
        
        # Find contours in binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get max contour
            max_contour = max(contours, key=cv2.contourArea)

            # Get bounding box
            x, y, w, h = cv2.boundingRect(max_contour)

            # # Draw bounding box
            # cv2.rectangle(rotated, (x, y), (x + w, y + h), (0, 0, 255), 5)

            # Crop image
            cropped = rotated[y + thickness : y + h - thickness, x + thickness : x + w - thickness]
        else:
            # If no contours, keep the rotated image
            cropped = rotated

        # Binary (2): Làm rõ mặt chữ, đường net
        binary = enhance_and_binarize(cropped)
        # binary = denoise_image(binary, method="nl_means")

        table = detect_table(binary)
        # kernel = np.ones((3, 3), np.uint8)
        # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        # binary = cv2.dilate(binary, kernel, iterations=1)

        return rotated, cropped, table

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