import re
from unidecode import unidecode


def process_text(input_text):
    """Cleans and normalizes the input text."""
    try:
        # Remove leading and trailing whitespaces
        text = input_text.strip()

        # Replace multiple spaces with a single space
        cleaned_text = re.sub(r"\s+", " ", text)

        # Convert text to lowercase and remove accents
        final_text = unidecode(cleaned_text.lower())

        return final_text
    except Exception as e:
        print(f"Error processing text: {e}")
        # Return an empty string if an error occurs
        return ""  


def process_output_old(ocr_output):

    try:
        order_info = []
        order_details = []
        order_summary = []

        lines = ocr_output.split("\n")

        for line in lines:

            is_summary = True

            # Remove empty lines
            line = line.strip()
            if not line:
                continue

            # Logic for detecting order details
            if "|" in line:
                order_details.append([col.strip() for col in line.split("|")])
                continue

            # Logic for detecting order information
            for token in [
                "hoa don ban hang", "shop", "hot line", "hotline", "nhan vien",
                "khach hang", "dia chi", "khu vuc", "thoi gian"
            ]:
                if process_text(line).startswith(token):
                    line = OCR_CORRECTION[token] + line[len(token):]
                    order_info.append(line)
                    is_summary = False
                    break

            # Logic for detecting order summary
            if is_summary:
                for token in [
                    "tong so luong",
                    "tong tien hang",
                    "chiet khau hoa don",
                    "chiet kheu hoa don",
                    "tong cong",
                    "nguoi ban hang"
                ]:
                    if process_text(line).startswith(token):
                        line = OCR_CORRECTION[token] + line[len(token):]
                        order_summary.append(line)
                        break
                else:
                    order_summary.append(line)

        print(order_info)
        print(order_details)
        print(order_summary)

        return order_info, order_details, order_summary
    except Exception as e:
        print(f"Error processing OCR output: {e}")
        return [], [], []


def process_output(ocr_output):

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
    order_details = {
        "product_name": [],
        "unit_price": [],
        "quantity": [],
    }
    order_summary = {
        "total_quantity": 0,
        "total_amount": 0,
        "discount": 0,
        "monetary": 0,
    }
    
    address_pattern = r"dia chi:\s*(.*?)\s*khu vuc:"
    # re.DOTALL to match multiple lines
    address_match = re.search(address_pattern, ocr_output, re.DOTALL)
    if address_match:
        profile_info["address"] = address_match.group(1).strip()
        ocr_output = re.sub(rf"{profile_info["address"]}", "", ocr_output).strip()

    lines = ocr_output.split("\n")
    for line in lines:
        line = process_text(line)
        # Remove empty lines
        if not line:
            continue
    # ocr_output = "\n".join(lines)
    
    for line in lines:
        if line.startswith("shop"):
            profile_info["shop_name"] = re.sub(rf"\bshop\b", "", line).strip()

        if line.startswith("hotline"):
            profile_info["hotline"] = re.sub(rf"\bhotline:\b", "", line).strip()

        if line.startswith("nhan vien ban hang"):
            profile_info["employee_name"] = re.sub(rf"\bnhan vien ban hang:\b", "", line).strip()

        if "sdt" in line:
            phone_pattern = r"\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b"
            phone_match = re.search(phone_pattern, line)
            if phone_match:
                profile_info["customer_phone"] = phone_match.group()
            tmp_line = re.sub(rf"\bsdt:\b", "", line).strip()    
            profile_info["customer_name"] = re.sub(rf"{profile_info["customer_phone"]}", "", tmp_line).strip()
        
        if line.startswith("khu vuc"):
            profile_info["region"] = re.sub(rf"\b=khu vuc:\b", "", line).strip()

        if line.startswith("thoi gian ban hang"):
            profile_info["employee_name"] = re.sub(rf"\bthoi gian giao hang:\b", "", line).strip()
        
    return profile_info, order_details, order_summary