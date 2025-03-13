import cv2
import numpy as np
from PIL import Image, ImageEnhance
import yaml 

# pip install git+https://github.com/pbcquoc/vietocr.git
# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent))
# print(sys.path)

from utils.models.vietocr.tool.predictor import Predictor
from utils.models.vietocr.model.vocab import Vocab
from utils.models.vietocr.tool.config import Cfg
from utils.models.vietocr.model.transformerocr import VietOCR


def detect_table(image):
    # Convert image to grayscale if it's not already
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Áp dụng adaptive threshold để làm nổi bật đường viền bảng
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

    # Kernel để nhận diện đường ngang (dòng kẻ ngang)
    kernel_hor = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_hor, iterations=2)

    # Kernel để nhận diện đường dọc (cột kẻ dọc)
    kernel_ver = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_ver, iterations=2)

    # Kết hợp cả hai để lấy toàn bộ đường viền bảng
    table_mask = cv2.add(horizontal_lines, vertical_lines)

    # Tìm contour của bảng
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lọc contour có diện tích lớn (tránh nhiễu nhỏ)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 50:  # Lọc bỏ vùng nhỏ
            table_roi = image[y:y+h, x:x+w]  # Cắt bảng
            break
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    table_roi = clahe.apply(table_roi)

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
        lines = [line for line in h_lines if thresh[1] - thin_thresh <= line[0][1] <= thresh[1] + thin_thresh]
        h_lines = [line for line in h_lines if thresh[1] - thin_thresh > line[0][1] or line[0][1] > thresh[1] + thin_thresh]
        
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
        lines = [line for line in v_lines if thresh[0] - thin_thresh <= line[0][0] <= thresh[0] + thin_thresh]
        v_lines = [line for line in v_lines if thresh[0] - thin_thresh > line[0][0] or line[0][0] > thresh[0] + thin_thresh]
        
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
    if image is None or image.size == 0:
        raise ValueError("Error: Input image is empty or not loaded correctly.")

    # Convert to grayscale if not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Detect horizontal and vertical lines
    kernel_hor = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    kernel_ver = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_hor, iterations=2)
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_ver, iterations=2)

    # Use Hough Transform to find lines
    h_lines = cv2.HoughLinesP(horizontal_lines, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
    v_lines = cv2.HoughLinesP(vertical_lines, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)

    # Ensure lines exist before processing
    if h_lines is None:
        h_lines = []
    else:
        h_lines = [line for line in h_lines]  # Đảm bảo danh sách đúng định dạng

    if v_lines is None:
        v_lines = []
    else:
        v_lines = [line for line in v_lines]  # Đảm bảo danh sách đúng định dạng

    # Extend lines
    h_lines = group_h_lines(h_lines, thin_thresh)
    v_lines = group_v_lines(v_lines, thin_thresh)

    # Find intersection points
    points = []
    for hline in h_lines:
        for vline in v_lines:
            intersection = seg_intersect(
                (hline[0], hline[1]), (hline[2], hline[3]), 
                (vline[0], vline[1]), (vline[2], vline[3])
            )
            if intersection:
                points.append(list(intersection))

    # Sort and detect table cells
    table_cells = []
    for point in points:
        left, top = point
        right_points = sorted([p for p in points if p[0] > left and p[1] == top], key=lambda x: x[0])
        bottom_points = sorted([p for p in points if p[1] > top and p[0] == left], key=lambda x: x[1])

        if right_points and bottom_points:
            right, bottom = right_points[0][0], bottom_points[0][1]
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
            table_cells.append([left, top, right, bottom])

    return table_cells


def load_model(config_path):
    """
    Load configuration from a YAML file and initialize the VietOCR model.

    Args:
        config_path (str or Path): Path to the YAML configuration file.

    Returns:
        tuple: (config, model, vocab) 
            - config (dict): VietOCR model configuration.
            - model (VietOCR): Initialized OCR model.
            - vocab (Vocab): Vocabulary object.
    """
    # Load configuration from the YAML file
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Modify additional parameters
    # config['predictor']['beamsearch'] = False

    # Initialize vocabulary
    vocab = Vocab(config['vocab'])

    # Initialize the VietOCR model
    model = VietOCR(len(vocab),
                    config['backbone'],
                    config['cnn'],
                    config['transformer'],
                    config['seq_modeling'])

    # Move the model to the appropriate device (CPU/GPU)
    model = model.to(config['device'])
    model.load_state_dict(torch.load(
        'models/vgg19_bn-c79401a0.pth', map_location=config['device']), strict=False)

    return model, config, vocab


def text_recognization(image, table_cells, imgH=32):

    config_path = "utils/models/config/vgg-seq2seq.yml"
    model, config, vocab = load_model(config_path=config_path)

    crop_img_list = []
    coordinate_list = []
    for x_left, y_top, x_right, y_bottom in table_cells:
        cropped = image[y_top : y_bottom, x_left : x_right]
        crop_img_list.append(cropped)
        coordinate_list.append((x_left + x_right) /2, (y_top + y_bottom) / 2)

    # load model ocr
    ocr_model = Predictor(model=model, config=config, vocab=vocab)
    set_bucket_thresh = config['set_bucket_thresh']

    # predict
    ocr_result = ocr_model.batch_predict(crop_img_list, set_bucket_thresh)
    final_result = list(zip(coordinate_list, ocr_result))

    return final_result

