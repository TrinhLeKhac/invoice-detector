import re
from unidecode import unidecode

# regext extract information
SHOP_NAME_NO_ACCENT_PATTERN = r"(?:shop)\s*(.*?)\s*\.*\s*(?:hot line)\s*:?"
HOTLINE_NO_ACCENT_PATTERN = r"(?:hot line)\s*:?\s*(.*?)\s*\.*\s*(?:nhan vien ban hang)\s*:?"
EMPLOYEE_NAME_NO_ACCENT_PATTERN = r"(?:nhan vien ban hang)\s*:?\s*(.*?)\s*\.*\s*(?:khach hang)\s*:?"
CUSTOMER_NAME_NO_ACCENT_PATTERN = r"(?:khach hang)\s*:?\s*(.*?)\s*\.*\s*(?:sdt)\s*:?"
CUSTOMER_PHONE_NO_ACCENT_PATTERN = r"(?:sdt)\s*:?\s*(.*?)\s*\.*\s*(?:dia chi)\s*:?"
ADDRESS_NO_ACCENT_PATTERN = r"(?:dia chi)\s*:?\s*(.*?)\s*\.*\s*(?:khu vuc)\s*:?"
REGION_NO_ACCENT_PATTERN = r"(?:khu vuc)\s*:?\s*(.*?)\s*\.*\s*(?:thoi gian giao hang)\s*:?"
SHIPPING_TIME_NO_ACCENT_PATTERN = r"(?:thoi gian giao hang)\s*:?\s*(.*?)\s*\.*\s*(?:ten)\s*:?"

TOTAL_QUANTITY_NO_ACCENT_PATTERN = r"tong so luong\s*:?\s*([\d,\.]+)"
TOTAL_AMOUNT_NO_ACCENT_PATTERN = r"tong tien hang\s*:?\s*([\d,\.]+)"
DISCOUNT_NO_ACCENT_PATTERN = r"chiet khau hoa don\s*:?\s*([\d,\.]+)"
MONETARY_NO_ACCENT_PATTERN = r"tong cong\s*:?\s*([\d,\.]+)"


def extract_information(target, no_accent_target, pattern):    
    match = re.search(pattern, no_accent_target, re.IGNORECASE)
    
    if match:
        start_idx = match.start(1)
        end_idx = match.end(1)
        information = target[start_idx:end_idx]
        
        # remain = re.sub(re.escape(information), "", target).strip()
        # no_accent_remain = re.sub(re.escape(unidecode(information)), "", no_accent_target).strip()
        # assert len(remain) == len(no_accent_remain)
        
        return information
    return ""


def extract_and_normalize_phone_numbers(text):
    # Bắt đầu bằng +084, +84 hoặc 0
    pattern = r"\b(?:\+084|\+84|0)(\d{9})\b"
    
    # Lấy danh sách số
    matches = re.findall(pattern, text)
    
    # Thay +84, +084 bằng 0
    normalized_numbers = ["0" + num for num in matches]  
    
    return normalized_numbers


def extract_name(text):
    # Loại bỏ số và ký tự đặc biệt, chỉ giữ lại chữ cái và khoảng trắng
    cleaned_text = re.sub(r"[^a-zA-ZÀ-ỹ\s]", "", text)
    
    # Chuẩn hóa khoảng trắng dư thừa
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    
    return cleaned_text


def extract_address(text):
    # Chỉ giữ lại chữ, số, khoảng trắng, dấu phẩy, dấu / và dấu gạch ngang
    cleaned_text = re.sub(r"[^a-zA-ZÀ-ỹ0-9,\-\/\s]", "", text)
    
    # Chuẩn hóa khoảng trắng dư thừa
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    
    return cleaned_text


def normalize_number(currency_str):
    # Loại bỏ dấu phẩy hoặc dấu chấm dùng làm phân cách hàng nghìn
    cleaned_number = re.sub(r"[,.]", "", currency_str)
    return int(cleaned_number)


def process_output(ocr_output):

    # Define default information
    profile_info = {
        "shop_name": "",
        "hotline": [],
        "employee_name": "",
        "customer_name": "",
        "customer_phone": "",
        "address": "",
        "region": "",
        "shipping_time": "",
    }
    order_details = [
            {
                "product_name": "Product A",
                "unit_price": 1000,
                "quantity": 1,
                "total_price": 1000,
            },
            {
                "product_name": "Product B",
                "unit_price": 1000,
                "quantity": 2,
                "total_price": 2000,
            },
            {
                "product_name": "Product C",
                "unit_price": 10000,
                "quantity": 3,
                "total_price": 30000,
            },
        ]

    order_summary = {
        "total_quantity": 0,
        "total_amount": 0,
        "discount": 0,
        "monetary": 0,
    }

    # Normalize OCR output
    target = re.sub(r"\s+", " ", ocr_output).strip()
    no_accent_target = unidecode(target)
    assert len(target) == len(no_accent_target)

    # Extract information
    shop_name = extract_information(target, no_accent_target, SHOP_NAME_NO_ACCENT_PATTERN)
    shop_name = extract_name(shop_name)
    profile_info["shop_name"] = shop_name

    hotline = extract_information(target, no_accent_target, HOTLINE_NO_ACCENT_PATTERN)
    hotline =  extract_and_normalize_phone_numbers(hotline)
    profile_info["hotline"] = hotline

    employee_name = extract_information(target, no_accent_target, EMPLOYEE_NAME_NO_ACCENT_PATTERN)
    employee_name = extract_name(employee_name)
    profile_info["employee_name"] = employee_name

    customer_name = extract_information(target, no_accent_target, CUSTOMER_NAME_NO_ACCENT_PATTERN)
    customer_name = extract_name(customer_name)
    profile_info["customer_name"] = customer_name

    customer_phone = extract_information(target, no_accent_target, CUSTOMER_PHONE_NO_ACCENT_PATTERN)
    customer_phone =  extract_and_normalize_phone_numbers(customer_phone)[0]
    profile_info["customer_phone"] = customer_phone
    
    address = extract_information(target, no_accent_target, ADDRESS_NO_ACCENT_PATTERN)
    address = extract_address(address)
    profile_info["address"] = address

    region = extract_information(target, no_accent_target, REGION_NO_ACCENT_PATTERN)
    region = extract_name(region)
    profile_info["region"] = region

    shipping_time = extract_information(target, no_accent_target, SHIPPING_TIME_NO_ACCENT_PATTERN)
    profile_info["shipping_time"] = shipping_time

    total_quantity = extract_information(target, no_accent_target, TOTAL_QUANTITY_NO_ACCENT_PATTERN)
    total_quantity =  normalize_number(total_quantity)
    order_summary["total_quantity"] = total_quantity

    total_amount = extract_information(target, no_accent_target, TOTAL_AMOUNT_NO_ACCENT_PATTERN)
    total_amount =  normalize_number(total_amount)
    order_summary["total_amount"] = total_amount

    discount = extract_information(target, no_accent_target, DISCOUNT_NO_ACCENT_PATTERN)
    discount =  normalize_number(discount)
    order_summary["discount"] = discount
    
    monetary = extract_information(target, no_accent_target, MONETARY_NO_ACCENT_PATTERN)
    monetary =  normalize_number(monetary)
    order_summary["monetary"] = monetary

    return profile_info, order_details, order_summary