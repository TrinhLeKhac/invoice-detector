import re
from unidecode import unidecode

from utils.address.parser import parse_address

from utils.text.regex import *
from utils.text.helper import (
    clean_text_before_unidecode,
    extract_information,
    extract_name,
    normalize_number,
    normalize_datetime,
    extract_and_normalize_phone_numbers,
    normalize_name_by_weight,
    extract_address,
    validate_and_fill_amounts,
)

from utils.address.regex import *
from data.district import DISTRICT_DICTIONARY
from data.province import PROVINCE_DICTIONARY
from data.ward import WARD_DICTIONARY

from data.name import (
    FIRST_NAMES_DICT,
    FIRST_NAMES_SET,
    LAST_NAMES_DICT,
    LAST_NAMES_SET,
    MIDDLE_NAMES_DICT,
    MIDDLE_NAMES_SET,
)


def handle_general_information(general_information):

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

    order_summary = {
        "total_quantity": 0,
        "total_amount": 0,
        "discount": 0,
        "monetary": 0,
    }

    # Normalize OCR output
    target = clean_text_before_unidecode(general_information)
    no_accent_target = unidecode(target)
    # assert len(target) == len(no_accent_target)
    # print(target)
    # print(no_accent_target)

    # Extract information
    created_time = extract_information(target, no_accent_target, CREATED_TIME_PATTERN)
    created_time = normalize_datetime(created_time)
    profile_info["created_time"] = created_time

    shop_name = extract_information(target, no_accent_target, SHOP_NAME_PATTERN)
    shop_name = extract_name(shop_name)
    profile_info["shop_name"] = shop_name

    hotline = extract_information(target, no_accent_target, HOTLINE_PATTERN)
    lst_hotline = extract_and_normalize_phone_numbers(hotline)
    profile_info["hotline"] = lst_hotline

    employee_name = extract_information(target, no_accent_target, EMPLOYEE_NAME_PATTERN)
    employee_name = extract_name(employee_name)
    employee_name = normalize_name_by_weight(
        employee_name, FIRST_NAMES_DICT, MIDDLE_NAMES_DICT, LAST_NAMES_DICT
    )
    profile_info["employee_name"] = employee_name

    customer_name = extract_information(target, no_accent_target, CUSTOMER_NAME_PATTERN)
    customer_name = extract_name(customer_name)
    customer_name = normalize_name_by_weight(
        customer_name, FIRST_NAMES_DICT, MIDDLE_NAMES_DICT, LAST_NAMES_DICT
    )
    profile_info["customer_name"] = customer_name

    customer_phone = extract_information(
        target, no_accent_target, CUSTOMER_PHONE_PATTERN
    )
    lst_customer_phone = extract_and_normalize_phone_numbers(customer_phone)
    if len(lst_customer_phone) > 0:
        profile_info["customer_phone"] = lst_customer_phone[0]

    address = extract_information(target, no_accent_target, ADDRESS_PATTERN)
    address = extract_address(address)
    address = parse_address(
        address,
        PROVINCE_DICTIONARY,
        DISTRICT_DICTIONARY,
        WARD_DICTIONARY,
        SPECIAL_ENDING,
    )
    profile_info["address"] = address

    region = extract_information(target, no_accent_target, REGION_PATTERN)
    region = extract_name(region)
    profile_info["region"] = region

    shipping_time = extract_information(target, no_accent_target, SHIPPING_TIME_PATTERN)
    shipping_time = normalize_datetime(shipping_time)
    profile_info["shipping_time"] = shipping_time

    total_quantity = extract_information(
        target, no_accent_target, TOTAL_QUANTITY_PATTERN, direct=True
    )
    total_quantity = normalize_number(total_quantity)
    order_summary["total_quantity"] = total_quantity

    total_amount = extract_information(
        target, no_accent_target, TOTAL_AMOUNT_PATTERN, direct=True
    )
    total_amount = normalize_number(total_amount)

    discount = extract_information(
        target, no_accent_target, DISCOUNT_PATTERN, direct=True
    )
    discount = normalize_number(discount)

    monetary = extract_information(
        target, no_accent_target, MONETARY_PATTERN, direct=True
    )
    monetary = normalize_number(monetary)

    total_amount, discount, monetary = validate_and_fill_amounts(
        total_amount, discount, monetary
    )
    order_summary["total_amount"] = total_amount
    order_summary["discount"] = discount
    order_summary["monetary"] = monetary

    return profile_info, order_summary


def handle_table_information(raw_table_information):

    def normalize_column_name(column_name):
        """Normalize column names using regex matching"""
        for pattern, standard_name in TABLE_COLUMN_MAPPING.items():
            if re.search(pattern, column_name, re.IGNORECASE | re.DOTALL):
                return standard_name
        return column_name

    # Default row if extracted data is missing
    default_table_information = {
        "product_name": "",
        "quantity": 0,
        "unit_price": 0,
        "total_price": 0,
    }

    # If raw_table_information is empty, return default values
    if not raw_table_information:
        return [default_table_information]

    table_information = []
    for raw_table_row in raw_table_information:
        normalized_row = {}

        # Normalize column names and values
        for column, value in raw_table_row.items():
            normalized_column = normalize_column_name(column)

            if normalized_column in ["quantity", "unit_price", "total_price"]:
                value = normalize_number(value)

            if normalized_column in [
                "product_name",
                "quantity",
                "unit_price",
                "total_price",
            ]:
                normalized_row[normalized_column] = value

        # Fill missing values with defaults
        for key, default_value in default_table_information.items():
            if key not in normalized_row:
                normalized_row[key] = default_value

        # If only total_price is present, assume quantity = 1
        if normalized_row["total_price"] > 0 and (
            normalized_row["quantity"] == 0 or normalized_row["unit_price"] == 0
        ):
            normalized_row["quantity"] = 1
            normalized_row["unit_price"] = normalized_row["total_price"]

        # Ensure quantity * unit_price = total_price
        expected_total_price = normalized_row["quantity"] * normalized_row["unit_price"]

        if normalized_row["total_price"] != expected_total_price:
            if normalized_row["total_price"] % normalized_row["unit_price"] != 0:
                # If total_price is not divisible by unit_price, update total_price
                normalized_row["total_price"] = expected_total_price
            else:
                # Otherwise, update quantity to match total_price
                normalized_row["quantity"] = (
                    normalized_row["total_price"] // normalized_row["unit_price"]
                )

        table_information.append(normalized_row)

    return table_information
