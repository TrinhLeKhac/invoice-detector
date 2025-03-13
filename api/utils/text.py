import re
from unidecode import unidecode
import itertools
from datetime import datetime

# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent))

from utils.address import *
from utils.data.province import PROVINCE_DICTIONARY
from utils.data.district import DISTRICT_DICTIONARY
from utils.data.ward import WARD_DICTIONARY

from utils.data.name import FIRST_NAMES_SET, MIDDLE_NAMES_SET, LAST_NAMES_SET, FIRST_NAMES_DICT, MIDDLE_NAMES_DICT, LAST_NAMES_DICT

# Regex patterns to extract information

SPECIAL_CHARACTER = r"=><„,.:;“”\/-_(){}|?"
DIGIT_AND_MISSPELLED_CHAR = "\dOÔƠ,\."
OPTIONAL_COLON = "\s*:?\s*"
VOCAB = r'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '

# Regex to capture various datetime formats
TIME_DATE_PATTERN = r"(?:(\d{2}[:]\d{2}(?::\d{2})?)\s+)?(\d{2}[\s/-]\d{2}[\s/-]\d{4})(?:\s+(\d{2}[:]\d{2}(?::\d{2})?))?"

CREATED_TIME_PATTERN = r"(?:(?:hoa don ban hang)[^0-9]*)?(\d{1,2}\s*[/:.-]\s*\d{1,2}\s*[/:.-]\s*\d{4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?|\d{1,2}:\d{2}(?::\d{2})?\s+\d{1,2}\s*[/:.-]\s*\d{1,2}\s*[/:.-]\s*\d{4})"
SHOP_NAME_PATTERN = rf"(?:s\s*hop){OPTIONAL_COLON}(.*?)(?:hot\s*line)"
HOTLINE_PATTERN = rf"(?:hot\s*line){OPTIONAL_COLON}(.*?)(?:nhan\s*vien\s*ban\s*hang)"
EMPLOYEE_NAME_PATTERN = rf"(?:nhan\s*vien\s*ban\s*hang){OPTIONAL_COLON}(.*?)(?:khach\s*hang)"
CUSTOMER_NAME_PATTERN = rf"(?:khach\s*hang){OPTIONAL_COLON}(.*?)(?:s\s*dt)"
CUSTOMER_PHONE_PATTERN = rf"(?:s\s*dt){OPTIONAL_COLON}(.*?)(?:dia\s*chi)"

ADDRESS_PATTERN = rf"(?:dia\s*chi){OPTIONAL_COLON}(.*?)(?:khu\s*vuc)"
REGION_PATTERN = rf"(?:khu\s*vuc){OPTIONAL_COLON}(.*?)(?:thoi\s*gian\s*giao\s*hang)"
SHIPPING_TIME_PATTERN = rf"(?:thoi\s*gian\s*giao\s*hang){OPTIONAL_COLON}(.*?)(?:ten)"

# TOTAL_QUANTITY_PATTERN = rf"tong\s*so\s*luong{OPTIONAL_COLON}([{DIGIT_AND_MISSPELLED_CHAR}]+)"
# TOTAL_AMOUNT_PATTERN = rf"tong\s*tien\s*hang{OPTIONAL_COLON}([{DIGIT_AND_MISSPELLED_CHAR}]+)"
# DISCOUNT_PATTERN = rf"chiet\s*khau\s*hoa\s*don{OPTIONAL_COLON}([{DIGIT_AND_MISSPELLED_CHAR}]+)"
# MONETARY_PATTERN = rf"tong\s*cong{OPTIONAL_COLON}([{DIGIT_AND_MISSPELLED_CHAR}]+)"

TOTAL_QUANTITY_PATTERN = rf"t[ôoòóỏọõốồổỗộơờớởỡợ]ng\s*s[ôoòóỏọõốồổỗộơờớởỡợ]\s*l[uưừứửữự][oôọóòõốồổỗộơờớởỡợ]ng{OPTIONAL_COLON}([{DIGIT_AND_MISSPELLED_CHAR}]+)"
TOTAL_AMOUNT_PATTERN = rf"t[ôoòóỏọõốồổỗộơờớởỡợ]ng\s*t[iìíĩịîï][êeéèẽẹêềếểễệ]n\s*h[aàáãạâấầẩẫậăắằẳẵặ]ng{OPTIONAL_COLON}([{DIGIT_AND_MISSPELLED_CHAR}]+)"
DISCOUNT_PATTERN = rf"ch[iìíĩịîï][êeéèẽẹêềếểễệ]t\s*kh[aàáãạâấầẩẫậăắằẳẵặ][uưừứửữự]\s*h[oòóỏọõôốồổỗộơờớởỡợ][aàáãạâấầẩẫậăắằẳẵặ]\s*[đd][oòóỏọõôốồổỗộơờớởỡợ]n{OPTIONAL_COLON}([{DIGIT_AND_MISSPELLED_CHAR}]+)"
MONETARY_PATTERN = rf"t[ôoòóỏọõốồổỗộơờớởỡợ]ng\s*c[oôòóỏọõốồổỗộơờớởỡợ]ng{OPTIONAL_COLON}([{DIGIT_AND_MISSPELLED_CHAR}]+)"


def clean_text_before_unidecode(text):

    """
    Cleans the text by removing:
    - Special characters that may expand (e.g., œ -> oe)
    - Emojis
    - Invisible whitespace characters (\u200b, \u00A0)
    - Soft hyphen (\u00AD)
    """

    # # Remove special characters (keeps only letters, digits, and spaces)
    # text = re.sub(r'[^A-Za-zÀ-ỹ0-9\s]', '', text)

    # Remove emojis using regex (matches all Unicode emoji ranges)
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]', '', text)
    
    # Remove invisible whitespace characters and soft hyphen
    text = re.sub(r'[\u200b\u00A0\u00AD]', '', text)
    
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_information(target, no_accent_target, pattern, direct=False):    

    """
    Extracts the desired information from `target` using regex on `no_accent_target`.

    This function applies the regex pattern to `no_accent_target`, then retrieves 
    the corresponding substring from `target` based on the matched index range.

    Args:
        target (str): The original text (with accents).
        no_accent_target (str): The text without accents, used for regex matching.
        pattern (str): The regex pattern for extraction.

    Returns:
        str: Extracted information if found, else an empty string.
    """

    if direct:
        match = re.search(pattern, target, re.IGNORECASE)
    
        if match:
            return match.group(1)  # Capture the first group
        return ""
    else:
        match = re.search(pattern, no_accent_target, re.IGNORECASE)
        
        if match:
            # Get the start and end indices of the captured group in `no_accent_target`
            start_idx = match.start(1)
            end_idx = match.end(1)

            # Extract the corresponding substring from the original `target`
            information = target[start_idx:end_idx]
            
            return information
        return ""


def extract_and_normalize_phone_numbers(text):

    """
    Extracts and normalizes phone numbers from a given text.

    This function searches for phone numbers that:
    - Start with "+084", "+84", or "0"
    - Are followed by exactly **9 or 10 digits**

    The extracted numbers are normalized by:
    - Removing "+084" and "+84", replacing them with "0"
    - Ensuring all phone numbers are returned in a standard "0XXXXXXXXX(X)" format

    Args:
        text (str): The input string containing phone numbers.

    Returns:
        list: A list of normalized phone numbers in "0XXXXXXXXX(X)" format.
    # Remove numbers and special characters, keeping only letters and spaces
    """

    # Matches phone numbers starting with +084, +84, or 0 and followed by 9 or 10 digits
    pattern = r"\b(?:\+084|\+84|0)(\d{9,10})\b"
    
    # Get list of matched numbers
    matches = re.findall(pattern, text)
    
    # Replace +84, +084 with 0
    normalized_numbers = ["0" + num for num in matches]  
    
    return normalized_numbers


def extract_name(text):

    """
    Extracts and cleans a name from a given text by removing unwanted characters.

    This function performs the following steps:
    - Removes special characters**, keeping only letters (both English and accented characters) and spaces.
    - Normalizes spaces**, ensuring there are no extra or trailing spaces.

    Args:
        text (str): The input string containing a name.

    Returns:
        str: A cleaned name with only valid characters and properly formatted spacing.
    """

    # Remove special characters, keeping only letters (including accents) and spaces
    cleaned_text = re.sub(r"[^a-zA-ZÀ-ỹ\s]", "", text)
    
    # Normalize spaces (convert multiple spaces to a single space)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    
    return cleaned_text


def normalize_name(name: str, first_names: set, middle_names: set, last_names: set) -> str:
    """
    Normalize a Vietnamese name by correcting misspellings, capitalizing words, 
    and rearranging components in the correct order: Last Name - Middle Name(s) - First Name.

    Args:
        name (str): The input name to be normalized.
        first_names (set): A set of valid last names (Họ).
        middle_names (set): A set of valid middle names (Chữ lót).
        last_names (set): A set of valid first names (Tên).

    Returns:
        str: The normalized full name or an empty string if no valid name is found.
    """

    def find_best_match(word, valid_set):
        """Find the best match for a word in the given set by comparing its non-accented version."""
        candidates = [vn_name for vn_name in valid_set if unidecode(vn_name).lower() == unidecode(word).lower()]
        return candidates[0] if candidates else None

    words = name.split()
    words = [word.capitalize() for word in words]  # Capitalize the first letter of each word

    # Remove invalid words
    valid_words = []
    for word in words:
        if word in first_names or word in middle_names or word in last_names:
            valid_words.append(word)
        else:
            match = find_best_match(word, first_names | middle_names | last_names)
            if match:
                valid_words.append(match)

    if not valid_words:
        return ""

    first, middle, last = "", [], ""

    # Họ
    for word in valid_words:
        if word in first_names:
            first = word
            valid_words.remove(word)
            break
    else:  # If no exact match is found, try finding a non-accented match
        for word in valid_words:
            match = find_best_match(word, first_names)
            if match:
                first = match
                valid_words.remove(word)
                break

    # Tên
    for word in reversed(valid_words):
        if word in first_names.union(last_names):
            last = word
            valid_words.remove(word)
            break
    else:  # If no exact match is found, try finding a non-accented match
        for word in reversed(valid_words):
            match = find_best_match(word, first_names.union(last_names))
            if match:
                last = match
                valid_words.remove(word)
                break

    # The remaining words are middle names (Chữ lót)
    middle = valid_words

    return " ".join(itertools.chain([first], middle, [last])).strip()


def normalize_name_by_weight(name: str, first_names: dict, middle_names: dict, last_names: dict, debug=False) -> str:
    """
    Normalize a Vietnamese name by correcting misspellings, capitalizing words, 
    and rearranging components in the correct order: first_names (Họ) - middle_names (Chữ lót) - last_names (Tên).

    Args:
        name (str): The input name to be normalized.
        first_names (dict): A dictionary of valid first names (Họ) with priority scores.
        middle_names (dict): A dictionary of valid middle names (Chữ lót) with priority scores.
        last_names (dict): A dictionary of valid last names (Tên) with priority scores.

    Returns:
        str: The normalized full name or an empty string if no valid name is found.
    """

    def find_best_match(word, valid_dict):
        """Find the best match for a word in the given dictionary by comparing its non-accented version."""
        candidates = [vn_name for vn_name in valid_dict if unidecode(vn_name).lower() == unidecode(word).lower()]
        if candidates:
            return max(candidates, key=lambda x: valid_dict[x])  # Choose the one with the highest priority
        return None

    words = name.split()
    words = [word.capitalize() for word in words]  # Capitalize the first letter of each word

    # Remove invalid words and correct misspellings
    valid_words = []
    for word in words:
        if word in first_names or word in middle_names or word in last_names:
            valid_words.append(word)  # Keep valid words
        else:
            match = find_best_match(word, {**first_names, **middle_names, **last_names})  
            if match:
                valid_words.append(match)  # Replace misspelled words with best matches

    if not valid_words:
        return ""

    if debug:    
        print("Valid words: ", valid_words)
    first, middle, last = "", [], ""

    # **Step 1: Identify the Last Name (Họ)**
    # Find exact matches in the first_names dictionary
    found_first_names = [word for word in valid_words if word in first_names]
    if debug:
        print("found_first_names: ", found_first_names)
    
    # Find potential last names by checking for misspelled versions in first_names
    # found_first_matches = {find_best_match(word, first_names): word for word in valid_words if word not in found_first_names}
    found_first_matches = {
        match: word for word in valid_words if word not in found_first_names and (match := find_best_match(word, first_names))
    }
    if debug:
        print("found_first_matches: ", found_first_matches)
    
    
    # Combine both exact and corrected last name candidates
    all_first_candidates = found_first_names + list(found_first_matches.keys())
    if debug:
        print("all_matches: ", all_first_candidates)
    
    # Select the last name with the highest priority score
    if all_first_candidates:
        first = max(all_first_candidates, key=lambda x: first_names[x])
        if debug:
            print("Họ: ", first)
        if first in found_first_names:
            if debug:
                print("Remove from first_names")
            valid_words.remove(first)  # Remove the selected last name
        else:
            if debug:
                print("Remove from first_matches")
            valid_words.remove(found_first_matches[first])  # Remove the misspelled version
            
    # **Step 2: Identify the First Name (Tên)**
    # Find exact matches for first name (including cases where it appears in first_names)
    found_last_names = [word for word in reversed(valid_words) if word in {**first_names, **last_names}]
    if debug:
        print("found_last_names: ", found_last_names)

    # Select the first name with the highest priority
    if found_last_names:
        last = max(found_last_names, key=lambda x: {**first_names, **last_names}.get(x))
        if debug:
            print("Tên: ", last)
        valid_words.remove(last)
    else:
        if debug:
            print("Tên else")
        for word in reversed(valid_words):
            match = find_best_match(word, {**first_names, **last_names})  
            if match:
                last = match
                if debug:
                    print("Tên: ", last)
                valid_words.remove(word)
                break

    # **Step 3: Remaining words are treated as Middle Name(s) (Chữ lót)**
    middle = valid_words

    return " ".join(itertools.chain([first], middle, [last])).strip()



def extract_address(text):

    """
    Extracts and cleans an address by removing unwanted characters.

    This function ensures that:
    Only valid address characters are kept**, including:
       - Letters (both English and accented characters)
       - Numbers (0-9)
       - Spaces
       - Commas (,), slashes (/), and hyphens (-) (commonly used in addresses)
    Extra spaces are normalized** to ensure proper formatting.

    Args:
        text (str): The input string containing an address.

    Returns:
        str: A cleaned address with only valid characters and properly formatted spacing.
    """

    # Keep only letters, numbers, spaces, commas, slashes, and hyphens
    cleaned_text = re.sub(r"[^a-zA-ZÀ-ỹ0-9,\-\/\s]", "", text)
    
    # Normalize spaces (convert multiple spaces to a single space)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    
    return cleaned_text


def normalize_number(currency_str):
    """
    Normalize a currency string by:
    - Replacing characters 'O', 'Ô', 'Ơ' with '0'
    - Removing thousand separators (`,` or `.`)
    - Keeping decimal points if present
    - Converting the cleaned string to a float or int
    
    Returns:
        int if no decimal part, otherwise float.
        Returns -1 if the input is not a valid number.
    """
    if not currency_str:
        return -1  # Handle empty string case
    
    # Replace 'O', 'Ô', 'Ơ' (common OCR errors) with '0'
    currency_str = re.sub(r"[OÔƠ]", "0", currency_str, flags=re.IGNORECASE)

    # Remove all non-numeric characters except digits
    cleaned_number = re.sub(r"[^\d]", "", currency_str)

    # Check if it's a valid number format
    try:
        number = int(cleaned_number)
        return number
    except ValueError:
        return 0  # Return 0 for invalid inputs


def validate_and_fill_amounts(total_amount, discount, monetary):
    """
    Validates and fills missing values for total_amount, discount, and monetary 
    based on the condition: total_amount = discount + monetary.

    Conditions:
    - total_amount > 0
    - discount <= total_amount
    - If discount > total_amount, set discount = total_amount
    - If any of the three values are missing, compute it using the formula.
    """

    # Ensure discount is within valid range
    if discount > total_amount:
        discount = total_amount - monetary
    if monetary > total_amount:
        monetary = total_amount - discount

    if total_amount == 0:
        total_amount = discount + monetary
    elif discount == 0:
        discount = total_amount - monetary
    elif monetary == 0:
        monetary = total_amount - discount

    return total_amount, discount, monetary


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


# def test_normalize_datetime():
#     # Test cases
#     test_cases = [
#         "03/01/2025 18:10",
#         "03/01/2025 18:10:10",
#         "18:10:10 03/01/2025",
#         "03-01-2025 18:10",
#         "03-01-2025 18:10:10",
#         "18:10:10 03-01-2025",
#         "03-01-2025",
#         "03/01/2025",
#         "03  /  01  /  2025   18:10",
#         "18:10:10   03 - 01 - 2025",
#     ]
#     # Run test cases
#     for test in test_cases:
#         print(f"Input: {test}\nOutput: {normalize_datetime(test)}\n")


def process_general_information(general_information):

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
    lst_hotline =  extract_and_normalize_phone_numbers(hotline)
    profile_info["hotline"] = lst_hotline

    employee_name = extract_information(target, no_accent_target, EMPLOYEE_NAME_PATTERN)
    employee_name = extract_name(employee_name)
    employee_name = normalize_name_by_weight(employee_name, FIRST_NAMES_DICT, MIDDLE_NAMES_DICT, LAST_NAMES_DICT)
    profile_info["employee_name"] = employee_name

    customer_name = extract_information(target, no_accent_target, CUSTOMER_NAME_PATTERN)
    customer_name = extract_name(customer_name)
    customer_name = normalize_name_by_weight(customer_name, FIRST_NAMES_DICT, MIDDLE_NAMES_DICT, LAST_NAMES_DICT)
    profile_info["customer_name"] = customer_name

    customer_phone = extract_information(target, no_accent_target, CUSTOMER_PHONE_PATTERN)
    lst_customer_phone =  extract_and_normalize_phone_numbers(customer_phone)
    if len(lst_customer_phone) > 0:
        profile_info["customer_phone"] = lst_customer_phone[0]
    
    address = extract_information(target, no_accent_target, ADDRESS_PATTERN)
    address = extract_address(address)
    address = parse_address(address, PROVINCE_DICTIONARY, DISTRICT_DICTIONARY, WARD_DICTIONARY, SPECIAL_ENDING)
    profile_info["address"] = address

    region = extract_information(target, no_accent_target, REGION_PATTERN)
    region = extract_name(region)
    profile_info["region"] = region

    shipping_time = extract_information(target, no_accent_target, SHIPPING_TIME_PATTERN)
    shipping_time = normalize_datetime(shipping_time)
    profile_info["shipping_time"] = shipping_time

    total_quantity = extract_information(target, no_accent_target, TOTAL_QUANTITY_PATTERN, direct=True)
    total_quantity =  normalize_number(total_quantity)
    order_summary["total_quantity"] = total_quantity

    total_amount = extract_information(target, no_accent_target, TOTAL_AMOUNT_PATTERN, direct=True)
    total_amount =  normalize_number(total_amount)

    discount = extract_information(target, no_accent_target, DISCOUNT_PATTERN, direct=True)
    discount =  normalize_number(discount)
    
    monetary = extract_information(target, no_accent_target, MONETARY_PATTERN, direct=True)
    monetary =  normalize_number(monetary)

    total_amount, discount, monetary = validate_and_fill_amounts(total_amount, discount, monetary)
    order_summary["total_amount"] = total_amount
    order_summary["discount"] = discount
    order_summary["monetary"] = monetary

    return profile_info, order_summary


def process_details_information(details_information):
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
    return order_details