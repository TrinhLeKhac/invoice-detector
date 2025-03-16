import numpy as np
from pytesseract import pytesseract


def parse_general_information(image, config="--psm 6", lang="vie"):
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


def parse_table_information(table_image, table_cells, border=3):
    """
    Extract table data from the given image and cell coordinates.

    Args:
        table_image: The image containing the table.
        table_cells: A list of cell coordinates [x1, y1, x2, y2].
        border: Padding to remove border noise.

    Returns:
        A list of dictionaries representing the table data.
    """

    # Each cell in the table is represented by [x1, y1, x2, y2], where:
    # (x1, y1): top-left corner
    # (x2, y2): bottom-right corner

    if not table_cells:
        print("No information extracted from table")
        return []

    # Extract unique row start positions (y1) and column start positions (x1)
    rows = sorted(set(cell[1] for cell in table_cells))
    cols = sorted(set(cell[0] for cell in table_cells))
    num_cols = len(cols)

    if num_cols == 0 or len(rows) == 0:
        print("No columns or rows found")
        return []

    h, w = table_image.shape[:2]  # Get image dimensions
    table_data = []

    # Extract column headers
    headers = []
    for x in cols:
        header_cell = next((cell for cell in table_cells if cell[0] == x and cell[1] == rows[0]), None)
        if header_cell:
            x1, y1, x2, y2 = header_cell
            x1, y1, x2, y2 = max(0, x1 + border), max(0, y1 + border), min(w, x2 - border), min(h, y2 - border)
            if x2 > x1 and y2 > y1:
                header_image = table_image[y1:y2, x1:x2]
                header_text = pytesseract.image_to_string(header_image, config="--psm 6", lang="vie").strip()
                headers.append(header_text)
            else:
                headers.append("")
        else:
            headers.append("")
    
    # Extract table rows
    for row_idx in range(1, len(rows)):  # Skip header row
        row_data = {}
        for col_idx, x in enumerate(cols):
            cell = next((cell for cell in table_cells if cell[0] == x and cell[1] == rows[row_idx]), None)
            if cell:
                x1, y1, x2, y2 = cell
                x1, y1, x2, y2 = max(0, x1 + border), max(0, y1 + border), min(w, x2 - border), min(h, y2 - border)
                if x2 > x1 and y2 > y1:
                    cell_image = table_image[y1:y2, x1:x2]
                    cell_text = pytesseract.image_to_string(cell_image, config="--psm 6", lang="vie").strip()
                    row_data[headers[col_idx]] = cell_text
                else:
                    row_data[headers[col_idx]] = ""
            else:
                row_data[headers[col_idx]] = ""  # Empty cell if missing
        table_data.append(row_data)

    return table_data
