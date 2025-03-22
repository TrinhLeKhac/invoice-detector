import re

import cv2
import numpy as np
from pytesseract import pytesseract

from utils.image.helper import (denoise, deskew_image, detect_and_crop_invoice,
                                enhance_contrast, grayscale, invert,
                                lighten_text, remove_shadow, remove_tape,
                                thicken_text)
from utils.image.ocr_parser import parse_table_information
from utils.table.detector import detect_cells, detect_table
from utils.text.handler import handle_table_information
# from utils.table.model_detector import InvoiceTableDetector


def count_valid_fields(output):
    valid_fields = 0
    for entry in output:
        valid_fields += sum(1 for _, value in entry.items() if value not in ['', 0])
    return valid_fields

def merge_outputs(output1, output2):
    """Merge data from two outputs by pairing entries, prioritizing the entry with more valid fields."""
    merged_output = []
    
    max_length = max(len(output1), len(output2))

    for i in range(max_length):
        entry1 = output1[i] if i < len(output1) else {}
        entry2 = output2[i] if i < len(output2) else {}

        # Determine which entry has more valid fields
        if count_valid_fields([entry1]) > count_valid_fields([entry2]):
            base_entry = entry1.copy()
            other_entry = entry2
        # Prioritize entry2 in case of a tie
        else:
            base_entry = entry2.copy()
            other_entry = entry1
        
        # Fill in missing information from the other entry
        for key, value in other_entry.items():
            if key not in base_entry or base_entry[key] in [None, "", 0]:
                base_entry[key] = value
        
        # Calculate missing values if possible
        quantity = base_entry.get("quantity")
        unit_price = base_entry.get("unit_price")
        total_price = base_entry.get("total_price")

        if quantity in [None, "", 0] and unit_price and total_price:
            base_entry["quantity"] = int(total_price / unit_price)
        elif unit_price in [None, "", 0] and quantity and total_price:
            base_entry["unit_price"] = int(total_price / quantity)
        elif total_price in [None, "", 0] and quantity and unit_price:
            base_entry["total_price"] = int(quantity * unit_price)

        merged_output.append(base_entry)

    return merged_output

def pipeline_extract_table(image):
    # Remove shadow
    non_shadow = remove_shadow(image)

    # Convert to grayscale
    gray = grayscale(non_shadow)

    # Deskew image
    deskewed = deskew_image(gray, debug=True)

    # Detect table
    table_roi = detect_table(deskewed)

    return table_roi

def pipeline_extract_information(table_roi, border=3):
    # Detect cells
    table_cells = detect_cells(table_roi)

    # Extract information
    raw_table_information = parse_table_information(
        table_roi, table_cells, border=border
    )
    
    table_information = handle_table_information(raw_table_information, debug=False)

    return table_information

def processing_image(image, border=5, model=False):
    """Processes an image: converts to grayscale, rotates, binarizes, finds contours, and crops."""
    # Crop invoice
    cropped = detect_and_crop_invoice(image, correct=False)

    if (cropped.shape[0] == 0) or (cropped.shape[1] == 0):
        print("Warning: Unable to crop the invoice")
        cropped = image.copy()
    
    if model:
        detector = InvoiceTableDetector()
        table_roi = detector.detect_tables(cropped)
        if table_roi is not None:
            print("Table detected by model")
            table_roi_by_cropped = table_roi
            table_roi_by_extracted = table_roi
            table_information = pipeline_extract_information(table_roi)
        else:
            print("No table detected")
            table_roi_by_cropped = None
            table_roi_by_extracted = None
            table_information = ""
    else:
        table_roi_by_cropped = detect_table(cropped)
        table_roi_by_extracted = pipeline_extract_table(cropped)
        
        table_information_by_cropped = pipeline_extract_information(table_roi_by_cropped, border=border)
        table_information_by_extracted = pipeline_extract_information(table_roi_by_extracted, border=border)
        print("Information by cropping: ", table_information_by_cropped)
        print("Information by extracting: ", table_information_by_extracted)

        # The merge_outputs() function prioritizes table_information_by_extracted in case of a tie
        # product_name from extracted is less prone to font errors than from cropped
        table_information = merge_outputs(table_information_by_cropped, table_information_by_extracted)

    return cropped, table_roi_by_cropped, table_roi_by_extracted, table_information
