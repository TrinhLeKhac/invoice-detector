import re
from unidecode import unidecode
from tabulate import tabulate
from ocr_correction import OCR_CORRECTION


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


def process_output(ocr_output):

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


def show_output(order_info, order_details, order_summary):

    try:
        print("=== Part 1: Order Information ===\n")
        print("\n".join(order_info))

        print("\n=== Part 2: Order Details ===\n")
        if order_details:
            headers = order_details[0]
            details = order_details[1:]
            print(tabulate(details, headers=headers, tablefmt="grid"))

        print("\n=== Part 3: Invoice Totals ===\n")
        print("\n".join(order_summary))
    except Exception as e:
        print(f"Error displaying output: {e}")