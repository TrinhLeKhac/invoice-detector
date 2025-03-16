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


def parse_table_information(table_image, table_cells, border=3, threshold=5):
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

    # Group rows and columns using the threshold
    def group_positions(positions, threshold):
        """Group similar coordinate values within the given threshold."""
        positions = sorted(positions)
        grouped = [[positions[0]]]

        for pos in positions[1:]:
            if abs(pos - grouped[-1][-1]) <= threshold:
                grouped[-1].append(pos)
            else:
                grouped.append([pos])

        return [int(np.mean(group)) for group in grouped]

    # Extract unique y1 (row start) and x1 (column start) with threshold-based grouping
    rows = group_positions([cell[1] for cell in table_cells], threshold)
    cols = group_positions([cell[0] for cell in table_cells], threshold)
    num_cols = len(cols)

    if num_cols == 0 or len(rows) == 0:
        print("No columns or rows found")
        return []

    table_data = []

    # Extract column headers
    headers = []
    for i in range(num_cols):
        header_cell = next(
            (
                cell
                for cell in table_cells
                if abs(cell[0] - cols[i]) <= threshold
                and abs(cell[1] - rows[0]) <= threshold
            ),
            None,
        )
        if header_cell:
            header_image = table_image[
                header_cell[1] + border : header_cell[3] - border,
                header_cell[0] + border : header_cell[2] - border,
            ]
            header_text = pytesseract.image_to_string(
                header_image, config="--psm 6", lang="vie"
            ).strip()
            headers.append(header_text)

    # Extract table rows
    for row_idx in range(1, len(rows)):  # Skip header row
        row_data = {}
        for col_idx in range(num_cols):
            try:
                cell = next(
                    (
                        cell
                        for cell in table_cells
                        if abs(cell[0] - cols[col_idx]) <= threshold
                        and abs(cell[1] - rows[row_idx]) <= threshold
                    ),
                    None,
                )
                if cell:
                    cell_image = table_image[
                        cell[1] + border : cell[3] - border,
                        cell[0] + border : cell[2] - border,
                    ]
                    cell_text = pytesseract.image_to_string(
                        cell_image, config="--psm 6", lang="vie"
                    ).strip()
                    row_data[headers[col_idx]] = cell_text
                else:
                    row_data[headers[col_idx]] = ""  # Empty cell if missing
            except Exception as e:
                print(f"Error processing table cell: {e}")
        table_data.append(row_data)

    return table_data
