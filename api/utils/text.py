import re
from unidecode import unidecode
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils.address import *
from utils.data.province import PROVINCE_DICTIONARY
from utils.data.district import DISTRICT_DICTIONARY
from utils.data.ward import WARD_DICTIONARY

# Regex patterns to extract information
CREATED_TIME_NO_ACCENT_PATTERN = r"(?:hoa don ban hang)\s*:?\s*([\d/\s:]+)\s*\.*\s*(?:shop)\s*:?"
SHOP_NAME_NO_ACCENT_PATTERN = r"(?:shop)\s*:?\s*(.*?)\s*\.*\s*(?:hot line)\s*:?"
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

# Regex to capture various datetime formats
TIME_DATE_PATTERN = r"(?:(\d{2}[:\s]+\d{2}(?::\d{2})?)\s+)?(\d{2}[\s/-]+\d{2}[\s/-]+\d{4})(?:\s+(\d{2}[:\s]+\d{2}(?::\d{2})?))?"


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
    # Matches phone numbers starting with +084, +84, or 0
    pattern = r"\b(?:\+084|\+84|0)(\d{9})\b"
    
    # Get list of matched numbers
    matches = re.findall(pattern, text)
    
    # Replace +84, +084 with 0
    normalized_numbers = ["0" + num for num in matches]  
    
    return normalized_numbers


def extract_name(text):
    # Remove numbers and special characters, keeping only letters and spaces
    cleaned_text = re.sub(r"[^a-zA-ZÀ-ỹ\s]", "", text)
    
    # Normalize extra spaces
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    
    return cleaned_text


def extract_address(text):
    # Keep only letters, numbers, spaces, commas, slashes, and hyphens
    cleaned_text = re.sub(r"[^a-zA-ZÀ-ỹ0-9,\-\/\s]", "", text)
    
    # Normalize extra spaces
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    
    return cleaned_text


def normalize_number(currency_str):
    # Remove commas or periods used as thousand separators
    cleaned_number = re.sub(r"[,.]", "", currency_str)
    
    return int(cleaned_number)


def normalize_datetime(text):
    match = re.search(TIME_DATE_PATTERN, text)
    if match:
        time_part_1 = match.group(1)  # Case where the time appears before the date
        date_part = match.group(2)    # Date in DD/MM/YYYY or DD-MM-YYYY format
        time_part_2 = match.group(3)  # Case where the time appears after the date

        # Select the correct time part (prioritizing the one after the date)
        time_part = time_part_2 if time_part_2 else time_part_1
        time_part = time_part.strip() if time_part else "00:00:00"

        # Normalize the date separator (replace multiple spaces, slashes, or dashes with "/")
        date_part = re.sub(r"[\s/-]+", "/", date_part.strip())

        # Normalize the time separator (replace multiple spaces or colons with ":")
        time_part = re.sub(r"[\s:]+", ":", time_part.strip())

        # If only HH:MM is provided, append ":00" to complete the time format
        if len(time_part) == 5:
            time_part += ":00"

        datetime_str = f"{date_part} {time_part}"
        
        # Convert to datetime object
        # try:
        #     dt_obj = datetime.strptime(datetime_str, "%d/%m/%Y %H:%M:%S")
        #     return dt_obj.strftime("%d/%m/%Y %H:%M:%S")
        # except ValueError:
        #     return f"Invalid format: {datetime_str}"

        return datetime_str
    return ""


def test_normalize_datetime():
    # Test cases
    test_cases = [
        "03/01/2025 18:10",
        "03/01/2025 18:10:10",
        "18:10:10 03/01/2025",
        "03-01-2025 18:10",
        "03-01-2025 18:10:10",
        "18:10:10 03-01-2025",
        "03-01-2025",
        "03/01/2025",
        "03  /  01  /  2025   18:10",
        "18:10:10   03 - 01 - 2025",
    ]
    # Run test cases
    for test in test_cases:
        print(f"Input: {test}\nOutput: {normalize_datetime(test)}\n")


def process_output(ocr_output):

    # Define default information
    profile_info = {
        "created_time": "",
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
    created_time = extract_information(target, no_accent_target, CREATED_TIME_NO_ACCENT_PATTERN)
    created_time = normalize_datetime(created_time)
    profile_info["created_time"] = created_time
    
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
    address = parse_address(address, PROVINCE_DICTIONARY, DISTRICT_DICTIONARY, WARD_DICTIONARY, SPECIAL_ENDING)
    profile_info["address"] = address

    region = extract_information(target, no_accent_target, REGION_NO_ACCENT_PATTERN)
    region = extract_name(region)
    profile_info["region"] = region

    shipping_time = extract_information(target, no_accent_target, SHIPPING_TIME_NO_ACCENT_PATTERN)
    shipping_time = normalize_datetime(shipping_time)
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