import re

import cv2
import numpy as np
from pytesseract import pytesseract

from utils.image.helper import (
    grayscale,
    remove_shadow,
    deskew_image,
    denoise,
    enhance_text_edges,
)
from utils.image.ocr_parser import parse_table_information
from utils.text.handler import handle_table_information
from utils.table.detector import detect_cells, detect_table


def processing_image(image, border=5):
    """Processes an image: converts to grayscale, rotates, binarizes, finds contours, and crops."""
    try:

        # Remove shadow
        # adjusted_image = remove_shadow(image)

        # Enhance contrast
        # enhanced = enhance_contrast(image)

        # Convert to grayscale
        gray = grayscale(image)

        # Convert to binary image to FIND CONTOURS
        # The surrounding area (e.g., packaging, background) may have noise.
        # This ensures accurate contour detection, allowing OCR to focus on invoice information.
        # However, excessive noise may lead to over-cropping, causing loss of invoice information.
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # Find contours in binary image
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # Get largest contour (assumed to be the main region of interest)
            max_contour = max(contours, key=cv2.contourArea)

            # Get bounding box
            x, y, w, h = cv2.boundingRect(max_contour)

            # # Draw bounding box
            # cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 255), 5)

            # Crop image with a safe border
            cropped = gray[y + border : y + h - border, x + border : x + w - border]
        else:
            # No contours found, keep the original grayscale image
            cropped = gray

        # Remove shadows
        non_shadow = remove_shadow(cropped)

        # Deskew images
        deskewed = deskew_image(non_shadow)

        # Remove noise
        denoised = denoise(deskewed, kernel_size=(1, 1), iterations=1, ksize=1)

        # Enhance text-edge
        enhanced = enhance_text_edges(denoised)  # binary image

        # Binary image
        # binary = binarize(enhanced)

        # Detect table
        table_roi = detect_table(denoised)  # denoised | enhanced
        # table_roi = thick_font(table_roi)
        # table_roi = invert(table_roi)

        # Detect cells
        table_cells = detect_cells(table_roi)
        print(table_cells)

        # Extract information
        raw_table_information = parse_table_information(
            table_roi, table_cells, border=border
        )
        print(raw_table_information)

        table_information = handle_table_information(raw_table_information)
        print(table_information)

        return denoised, enhanced, table_roi, table_information

    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None, None, None
