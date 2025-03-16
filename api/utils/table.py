import cv2
import numpy as np
import pandas as pd
from collections import OrderedDict
from pytesseract import pytesseract

def detect_table(image):
    # Convert image to grayscale if it's not already
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to highlight table borders
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Kernel for detecting horizontal lines
    kernel_hor = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horizontal_lines = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, kernel_hor, iterations=2
    )

    # Kernel for detecting vertical lines
    kernel_ver = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_ver, iterations=2)

    # Combine both to get the full table structure
    table_mask = cv2.add(horizontal_lines, vertical_lines)

    # Find table contours
    contours, _ = cv2.findContours(
        table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    table_roi = None

    # Filter contours to remove small noise and detect a valid table
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 100 and h > 100:  # Ignore small regions
            table_roi = gray[y : y + h, x : x + w]  # Crop the detected table
            break

    # If no table is detected, return None
    if table_roi is None:
        print("No table detected")
    
    #  # Apply CLAHE to enhance contrast
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # table_roi = clahe.apply(table_roi)

    return table_roi


def group_h_lines(h_lines, thin_thresh):
    """
    Groups close horizontal lines to avoid redundant detections.

    Args:
        h_lines (list): List of detected horizontal lines, where each line is represented as [[x1, y1, x2, y2]].
        thin_thresh (int): Threshold to determine whether two lines are close enough to be merged.

    Returns:
        list: List of grouped horizontal lines in the format [x_min, y, x_max, y].
    """
    new_h_lines = []
    while len(h_lines) > 0:
        # Find the topmost line (smallest y coordinate)
        thresh = sorted(h_lines, key=lambda x: x[0][1])[0][0]

        # Group lines that are close in the y-axis
        lines = [
            line
            for line in h_lines
            if thresh[1] - thin_thresh <= line[0][1] <= thresh[1] + thin_thresh
        ]
        h_lines = [
            line
            for line in h_lines
            if thresh[1] - thin_thresh > line[0][1]
            or line[0][1] > thresh[1] + thin_thresh
        ]

        # Extend the horizontal line to cover all detected segments
        x = [point for line in lines for point in [line[0][0], line[0][2]]]
        x_min, x_max = min(x) - int(5 * thin_thresh), max(x) + int(5 * thin_thresh)
        new_h_lines.append([x_min, thresh[1], x_max, thresh[1]])
    return new_h_lines


def group_v_lines(v_lines, thin_thresh):
    """
    Groups close vertical lines to avoid redundant detections.

    Args:
        v_lines (list): List of detected vertical lines, where each line is represented as [[x1, y1, x2, y2]].
        thin_thresh (int): Threshold to determine whether two lines are close enough to be merged.

    Returns:
        list: List of grouped vertical lines in the format [x, y_min, x, y_max].
    """
    new_v_lines = []
    while len(v_lines) > 0:
        # Find the leftmost line (smallest x coordinate)
        thresh = sorted(v_lines, key=lambda x: x[0][0])[0][0]

        # Group lines that are close in the x-axis
        lines = [
            line
            for line in v_lines
            if thresh[0] - thin_thresh <= line[0][0] <= thresh[0] + thin_thresh
        ]
        v_lines = [
            line
            for line in v_lines
            if thresh[0] - thin_thresh > line[0][0]
            or line[0][0] > thresh[0] + thin_thresh
        ]

        # Extend the vertical line to cover all detected segments
        y = [point for line in lines for point in [line[0][1], line[0][3]]]
        y_min, y_max = min(y) - int(4 * thin_thresh), max(y) + int(4 * thin_thresh)
        new_v_lines.append([thresh[0], y_min, thresh[0], y_max])
    return new_v_lines


def seg_intersect(p1, p2, p3, p4):
    """
    Finds the intersection point of two line segments if they intersect.

    Args:
        p1, p2 (tuple): Endpoints of the first line segment.
        p3, p4 (tuple): Endpoints of the second line segment.

    Returns:
        tuple or None: (x, y) coordinates of the intersection point, or None if lines are parallel.
    """
    a1, a2 = np.array(p1), np.array(p2)
    b1, b2 = np.array(p3), np.array(p4)
    da, db, dp = a2 - a1, b2 - b1, a1 - b1

    def perp(a):
        return np.array([-a[1], a[0]])

    dap = perp(da)
    denom = np.dot(dap, db)
    if denom == 0:
        return None  # Parallel lines, no intersection

    num = np.dot(dap, dp)
    intersection = (num / denom.astype(float)) * db + b1
    return int(intersection[0]), int(intersection[1])


def detect_cells(image, thin_thresh=5):
    """
    Detects table cells in a binary image.

    Args:
        image (numpy.ndarray): Binary (grayscale) image containing a table.

    Returns:
        list: List of detected cell coordinates as [x1, y1, x2, y2].
    """
    if image is None:
        print("No cells detected")
        return []

    # Convert to grayscale if not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Detect horizontal and vertical lines
    kernel_hor = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    kernel_ver = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

    horizontal_lines = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, kernel_hor, iterations=2
    )
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_ver, iterations=2)

    # Use Hough Transform to find lines
    h_lines = cv2.HoughLinesP(
        horizontal_lines, 
        1,                  # Khoảng cách pixel giữa các đường trong không gian Hough
        np.pi / 180,        # Độ phân giải góc trong không gian Hough (1 độ)
        10,                 # Ngưỡng (threshold) số điểm ảnh thẳng hàng tối thiểu để xác định 1 đoạn thẳng
        minLineLength=30,   # Độ dài tối thiểu của một đoạn thẳng hợp lệ
        maxLineGap=15       # Khoảng cách tối đa giữa hai đoạn để nối thành một đường
    ) # 50, 30, 10
    v_lines = cv2.HoughLinesP(
        vertical_lines, 1, np.pi / 180, 10, minLineLength=30, maxLineGap=15
    )

    # Ensure lines exist before processing
    if h_lines is None:
        h_lines = []

    if v_lines is None:
        v_lines = []

    # Extend lines
    h_lines = group_h_lines(h_lines, thin_thresh)
    v_lines = group_v_lines(v_lines, thin_thresh)

    # Find intersection points
    points = []
    for hline in h_lines:
        for vline in v_lines:
            intersection = seg_intersect(
                (hline[0], hline[1]),
                (hline[2], hline[3]),
                (vline[0], vline[1]),
                (vline[2], vline[3]),
            )
            if intersection:
                points.append(list(intersection))

    # Sort and detect table cells
    table_cells = []
    for point in points:
        left, top = point
        right_points = sorted(
            [p for p in points if p[0] > left and p[1] == top], key=lambda x: x[0]
        )
        bottom_points = sorted(
            [p for p in points if p[1] > top and p[0] == left], key=lambda x: x[1]
        )

        if right_points and bottom_points:
            right, bottom = right_points[0][0], bottom_points[0][1]
            # cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
            table_cells.append([left, top, right, bottom])

    # Modify coordinate of cells
    def remove_close_values(coords, threshold=5):
        filtered = [coords[0]]
        for i in range(1, len(coords)):
            if coords[i] - filtered[-1] >= threshold:
                filtered.append(coords[i])
        return filtered 
    
    if table_cells:
        # Extract unique x and y coordinates
        column_x_coords = sorted(set(cell[0] for cell in table_cells).union(set(cell[2] for cell in table_cells)))
        row_y_coords = sorted(set(cell[1] for cell in table_cells).union(set(cell[3] for cell in table_cells)))

        # Remove close values
        adjusted_columns = remove_close_values(column_x_coords, threshold=5)
        adjusted_rows = remove_close_values(row_y_coords, threshold=5)

        # Calculate new cells based on adjusted grid
        new_table_cells = []
        for row_idx in range(len(adjusted_rows) - 1):
            for col_idx in range(len(adjusted_columns) - 1):
                x1, y1 = adjusted_columns[col_idx], adjusted_rows[row_idx]
                x2, y2 = adjusted_columns[col_idx + 1], adjusted_rows[row_idx + 1]
                new_table_cells.append([x1, y1, x2, y2])
        table_cells = new_table_cells

    return table_cells


def extract_table_information(table_image, table_cells, border=3, threshold=5):
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
        header_cell = next((cell for cell in table_cells if abs(cell[0] - cols[i]) <= threshold and abs(cell[1] - rows[0]) <= threshold), None)
        if header_cell:
            header_image = table_image[
                header_cell[1] + border : header_cell[3] - border,
                header_cell[0] + border : header_cell[2] - border
            ]
            header_text = pytesseract.image_to_string(header_image, config="--psm 6", lang="vie").strip()
            headers.append(header_text)

    # Extract table rows
    for row_idx in range(1, len(rows)):  # Skip header row
        row_data = {}
        for col_idx in range(num_cols):
            try:
                cell = next((cell for cell in table_cells if abs(cell[0] - cols[col_idx]) <= threshold and abs(cell[1] - rows[row_idx]) <= threshold), None)
                if cell:
                    cell_image = table_image[
                        cell[1] + border : cell[3] - border,
                        cell[0] + border : cell[2] - border
                    ]
                    cell_text = pytesseract.image_to_string(cell_image, config="--psm 6", lang="vie").strip()
                    row_data[headers[col_idx]] = cell_text
                else:
                    row_data[headers[col_idx]] = ""  # Empty cell if missing
            except Exception as e:
                print(f"Error processing table cell: {e}")
        table_data.append(row_data)

    return table_data
