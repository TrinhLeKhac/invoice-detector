import re

import cv2
import numpy as np
from pytesseract import pytesseract

from utils.image.helper import (denoise, deskew_image, detect_and_crop_invoice,
                                enhance_contrast, grayscale, lighten_text,
                                remove_shadow, thicken_text)
from utils.image.ocr_parser import parse_table_information
from utils.table.detector import detect_cells, detect_table
from utils.text.handler import handle_table_information


def processing_image(image, border=5):
    """Processes an image: converts to grayscale, rotates, binarizes, finds contours, and crops."""
    # try:

    # Crop invoice
    cropped = detect_and_crop_invoice(image)
    if (cropped.shape[0] == 0) or (cropped.shape[1] == 0):
        print("Warning: Unable to crop the invoice")
        cropped = image.copy()

    # Remove shadow
    non_shadow = remove_shadow(cropped)

    # Enhance contrast
    # enhanced = enhance_contrast(non_shadow)

    # Convert to grayscale
    gray = grayscale(non_shadow)

    # Deskew images
    deskewed = deskew_image(gray)

    # Detect table
    table_roi = detect_table(deskewed)

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

    return cropped, deskewed, table_roi, table_information

    # except Exception as e:
    #     print(f"Error processing image: {e}")
    #     return None, None, None, None
