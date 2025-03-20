import math

import cv2
import numpy as np
from skimage.filters import threshold_sauvola


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


def invert(image):
    """Inverts the colors of an image using bitwise NOT operation.

    Args:
        image (np.ndarray): Input image (grayscale or color).

    Returns:
        np.ndarray: Inverted image.
    """
    inverted_image = cv2.bitwise_not(image)
    return inverted_image


def angle_between_points(p1, p2):
    """
    Calculate the angle between two points relative to the horizontal axis.
    If the angle is greater than 45 degrees, it is considered vertical, and the function returns (90 - angle).
    Otherwise, it is considered horizontal.

    Args:
        p1 (tuple): First point (x, y)
        p2 (tuple): Second point (x, y)

    Returns:
        tuple: (corrected angle, direction ('horizontal' or 'vertical'))
    """
    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]
    angle = abs(math.degrees(math.atan2(delta_y, delta_x)))
    angle = min(180 - angle, angle)
    if angle > 45:
        return 90 - angle, "vertical"
    else:
        return angle, "horizontal"


def correct_polygon(approx_polygon, limit_angle=5):
    """
    Adjusts the polygon by correcting points that form angles greater than a given threshold.

    Args:
        approx_polygon (numpy array): Approximate polygon points.
        limit_angle (int, optional): The angle threshold above which corrections are applied. Defaults to 5.

    Returns:
        numpy array: Corrected polygon points.
    """
    corrected_polygon = []
    num_points = len(approx_polygon)

    for i in range(num_points):
        p1 = approx_polygon[i][0]
        p2 = approx_polygon[(i + 1) % num_points][0]

        angle, direction = angle_between_points(p1, p2)

        if angle > limit_angle:
            print(
                f"Detecting p1({p1[0]}, {p1[1]}) and p2({p2[0]}, {p2[1]}) have angle {angle}"
            )
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2

            alpha = angle / 2  # Adjust half of the deviation angle
            rad = math.radians(alpha)

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            if direction == "horizontal":
                new_x1 = int(mid_x - (dx / 2) * math.cos(rad))
                new_y1 = int(mid_y - (dy / 2) * math.sin(rad))
                new_x2 = int(mid_x + (dx / 2) * math.cos(rad))
                new_y2 = int(mid_y + (dy / 2) * math.sin(rad))
            else:  # vertical
                new_x1 = int(mid_x - (dx / 2) * math.sin(rad))
                new_y1 = int(mid_y - (dy / 2) * math.cos(rad))
                new_x2 = int(mid_x + (dx / 2) * math.sin(rad))
                new_y2 = int(mid_y + (dy / 2) * math.cos(rad))

            corrected_polygon.append([[new_x1, new_y1]])
            corrected_polygon.append([[new_x2, new_y2]])
        else:
            corrected_polygon.append([p1])

    return np.array(corrected_polygon, dtype=np.int32)


def detect_and_crop_invoice(image, threshold=10, eps=0.05, correct=True, limit_angle=5):
    """
    Detects and extracts the invoice from an image by identifying contours and applying corrections.

    Args:
        image (numpy array): Input image.
        threshold (int, optional): Padding threshold for cropping. Defaults to 10.
        eps (float, optional): Approximation accuracy for contour polygon. Defaults to 0.05.
        correct (bool, optional): Whether to apply polygon correction. Defaults to True.
        limit_angle (int, optional): Minimum angle to trigger correction. Defaults to 5.

    Returns:
        numpy array: Cropped invoice image.
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for white color (adjustable)
    lower_white = np.array([0, 0, 120], dtype=np.uint8)
    upper_white = np.array([180, 60, 240], dtype=np.uint8)

    # Create mask for white regions
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No invoice detected!")
        return None

    # Find largest contour (assumed to be the invoice)
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a polygon with multiple sides
    epsilon = eps * cv2.arcLength(largest_contour, True)
    approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

    corrected_polygon = approx_polygon
    if correct:
        corrected_polygon = correct_polygon(approx_polygon, limit_angle=limit_angle)

    # Create a mask for the polygon
    mask_poly = np.zeros_like(mask)
    cv2.fillPoly(mask_poly, [corrected_polygon], 255)

    removed_background_image = cv2.bitwise_and(image, image, mask=mask_poly)

    # Crop using bounding rectangle box
    x, y, w, h = cv2.boundingRect(corrected_polygon)

    # Add a vertical threshold to avoid the issue of tape covering the text
    cropped_image = removed_background_image[
        y - threshold : y + h + threshold, x : x + w
    ]

    return cropped_image


def remove_tape(image):
    """
    Detect the boundary of the receipt and remove yellow tape around it.
    """
    tmp_image = image.copy()
    hsv = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2HSV)

    # Define yellow color range
    lower_yellow = np.array([10, 110, 50])
    upper_yellow = np.array([35, 255, 255])

    # Create mask for yellow regions
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(
            tmp_image, (x, y), (x + w, y + h), (255, 255, 255), -1
        )  # Fill with white

    return tmp_image


def enhance_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Enhances the contrast of an image using CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        image: Input image (grayscale or color).
        clip_limit: Threshold for contrast clipping in CLAHE.
        tile_grid_size: Size of the grid for CLAHE.

    Returns:
        The contrast-enhanced image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(image.shape) == 2:  # Grayscale image
        enhanced_image = clahe.apply(image)
    else:  # Color image, apply CLAHE to the L channel in LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)  # Apply CLAHE only on the L channel
        enhanced_lab = cv2.merge((l, a, b))
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced_image


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


def lighten_text(image, kernel_size=(2, 2), iterations=1):
    image = cv2.bitwise_not(image)
    kernel = np.ones(kernel_size, np.uint8)
    image = cv2.erode(image, kernel=kernel, iterations=iterations)
    image = cv2.bitwise_not(image)
    return image


def thicken_text(image, kernel_size=(2, 2), iterations=1):
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


def deskew_image(image, debug=True):
    """
    Detect and correct the skew of an image.
    Ensures that the rotation does not flip the image sideways.
    """

    angle = get_rotation_angle(image)
    if debug:
        print("Deskew angle: ", angle)
    if abs(angle) < 0.1:  # If angle is too small, no rotation is needed
        return image

    corrected_image = rotate_image(image, angle)

    return corrected_image
