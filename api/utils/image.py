import base64
import re

import cv2
import numpy as np
from numpy import mean
from pytesseract import pytesseract
from skimage.filters import threshold_sauvola

from utils.table import detect_cells, detect_table


def convert_from_base64(full_base64_string):
    try:
        match = re.match(
            r"data:image/(?P<format>png|jpeg|jpg);base64,(?P<data>.+)",
            full_base64_string,
        )

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


def invert(image):
    """Inverts the colors of an image using bitwise NOT operation.

    Args:
        image (np.ndarray): Input image (grayscale or color).

    Returns:
        np.ndarray: Inverted image.
    """
    inverted_image = cv2.bitwise_not(image)
    return inverted_image


def grayscale(image):
    """Converts a color image to grayscale if it's not already.

    Args:
        image (np.ndarray): Input image (grayscale or BGR color).

    Returns:
        np.ndarray: Grayscale image.
    """

    # Check if the image is already grayscale (2D array)
    if len(image.shape) == 2:
        gray = image
    else:
        # Convert BGR to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray


def denoise(image, kernel_size=(1, 1), iterations=1, ksize=3):
    """Removes noise from an image using morphological operations and median filtering.

    Args:
        image (np.ndarray): Input image (grayscale or binary).
        kernel_size (tuple): Size of the kernel used for morphological operations.
        iterations (int): Number of times dilation and erosion are applied.

    Returns:
        np.ndarray: The processed image with reduced noise.
    """

    # Validate input image
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a valid numpy array.")

    # Create a kernel (structuring element) for morphological operations
    kernel = np.ones(kernel_size, np.uint8)

    # Apply dilation (expands bright areas) to remove small black noise
    cleaned = cv2.dilate(image, kernel, iterations=iterations)

    # Apply erosion (shrinks bright areas) to remove small white noise
    cleaned = cv2.erode(cleaned, kernel, iterations=iterations)

    # Apply morphological closing (dilation followed by erosion) to close small holes in objects
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # Apply median blur to remove salt-and-pepper noise
    cleaned = cv2.medianBlur(cleaned, ksize=ksize)

    return cleaned


def thin_font(image, kernel_size=(2, 2), iterations=1):
    image = cv2.bitwise_not(image)
    kernel = np.ones(kernel_size, np.uint8)
    image = cv2.erode(image, kernel=kernel, iterations=iterations)
    image = cv2.bitwise_not(image)
    return image


def thick_font(image, kernel_size=(2, 2), iterations=1):
    image = cv2.bitwise_not(image)
    kernel = np.ones(kernel_size, np.uint8)
    image = cv2.dilate(image, kernel=kernel, iterations=iterations)
    image = cv2.bitwise_not(image)
    return image


def get_rotation_angle(image, angle_threshold=15):
    """
    Find the skew angle of the image using Hough Line Transform.
    This method detects straight lines and estimates the median rotation angle.
    """
    # Convert image to grayscale if it's not already
    gray = grayscale(image)

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
    rotated_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return rotated_image


def deskew_image(image):
    """
    Detect and correct the skew of an image.
    Ensures that the rotation does not flip the image sideways.
    """

    angle = get_rotation_angle(image)
    print(angle)
    if abs(angle) < 0.1:  # If angle is too small, no rotation is needed
        return image

    corrected_image = rotate_image(image, angle)

    return corrected_image


def binarize(image):
    """
    Enhances brightness using Adaptive Gamma Correction and applies adaptive binarization.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Binary image after enhancement.
    """

    # Convert image to grayscale if it's not already
    gray = grayscale(image)

    # ---- Step 1: Adaptive Gamma Correction ---- Ạdjust brightness based on Gamma
    mean_intensity = np.mean(gray)
    gamma = 1.5 if mean_intensity < 100 else 1.0
    gamma_corrected = np.power(gray / 255.0, gamma) * 255.0
    gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)

    # ---- Step 2: Adaptive Thresholding ----  Apply local threshold
    binary_adaptive = cv2.adaptiveThreshold(
        gamma_corrected,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41,
        5,  # Fine-tune blockSize & C
    )

    # ---- Step 3: Sauvola Binarization (Advanced) ---- Giúp giữ chi tiết tốt hơn trong vùng có độ tương phản thấp
    window_size = 15  # Window size for local thresholding
    thresh_sauvola = threshold_sauvola(gamma_corrected, window_size=window_size)
    binary_sauvola = (gamma_corrected > thresh_sauvola).astype(np.uint8) * 255

    # Combine the best binarization results (choose based on testing)
    final_binary = cv2.bitwise_and(binary_adaptive, binary_sauvola)

    return final_binary


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


def extract_information_from_table(table_image, table_cells):

    table_information = []
    for cell in table_cells:
        cell_x_min, cell_y_min, cell_x_max, cell_y_max = cell
        cell_image = table_image[cell_y_min:cell_y_max, cell_x_min:cell_x_max]

        table_information.append(extract_information_from_image(cell_image))

    return table_information


def remove_shadow(image):

    rgb_planes = cv2.split(image)

    result_planes = []
    result_threshold_planes = []
    for plane in rgb_planes:
        # Dilate the image, in order to get rid of the text
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))

        # Median blur the result with a decent sized kernel to further suppress any text.
        bg_img = cv2.medianBlur(dilated_img, 21)

        # Since we want black on white, we invert the result
        diff_img = 255 - cv2.absdiff(plane, bg_img)

        # Normalize the image, so that we use the full dynamic range.
        norm_img = diff_img.copy()  # Needed for 3.x compatibility
        cv2.normalize(
            diff_img,
            norm_img,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8UC1,
        )

        # At this point we still have the paper somewhat gray. We can truncate that away, and re-normalize the image.
        _, thr_img = cv2.threshold(norm_img, 230, 0, cv2.THRESH_TRUNC)
        cv2.normalize(
            thr_img,
            thr_img,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8UC1,
        )

        result_planes.append(diff_img)
        result_threshold_planes.append(thr_img)

    result = cv2.merge(result_planes)
    result_thresh = cv2.merge(result_threshold_planes)

    return result_thresh


def enhance_text_edges(image):

    w = image.shape[1]
    h = image.shape[0]
    w1 = int(w * 0.05)
    w2 = int(w * 0.95)
    h1 = int(h * 0.05)
    h2 = int(h * 0.95)
    ROI = image[h1:h2, w1:w2]  # 95% of center of the image
    threshold = np.mean(ROI) * 0.98  # 98% of average brightness

    gray = grayscale(image)
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)
    edged = 255 - cv2.Canny(blurred, 100, 150, apertureSize=7)

    thresh, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    return binary


def process_image(image, thickness=0):
    """Processes an image: converts to grayscale, rotates, binarizes, finds contours, and crops."""
    # try:

    # Remove shadow
    # adjusted_image = remove_shadow(image)

    # Convert to grayscale
    gray = grayscale(image)

    gray_cp = gray.copy()

    # Convert to binary image to FIND CONTOUR
    # Binary (1): Vùng xung quanh (thùng hàng, ..) bị lem
    # => contours chính xác => crop OCR focus được vào information
    # Tuy nhiên có lúc bị lem nhiều => cắt quá đà, bị mất thông tin
    _, binary = cv2.threshold(gray_cp, 180, 255, cv2.THRESH_BINARY)

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
        cropped = gray[
            y + thickness : y + h - thickness, x + thickness : x + w - thickness
        ]  # rotated
    else:
        # If no contours, keep the rotated image
        cropped = gray

    # Remove bóng
    non_shadow = remove_shadow(cropped)

    # Xoay hình
    deskewed = deskew_image(non_shadow)

    # Remove noise
    denoised = denoise(deskewed, kernel_size=(1, 1), iterations=1, ksize=1)

    # enhance text-edge
    enhanced = enhance_text_edges(denoised)  # binary image

    # Binary image
    # binary = binarize(enhanced)

    # Extract table
    table_roi = detect_table(enhanced)  # denoised | enhanced
    # table_roi = thick_font(table_roi)
    # table_roi = invert(table_roi)

    table_cells = detect_cells(table_roi)
    print(table_cells)

    details_information = extract_information_from_table(table_roi, table_cells)
    print(details_information)

    return denoised, enhanced, table_roi

    # except Exception as e:
    #     print(f"Error processing image: {e}")
    #     # Return None values if an error occurs
    #     return None, None, None, None
