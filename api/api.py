from flask import Flask, request, jsonify

from utils.image import extract_information_from_image, process_image
from utils.text import process_output

app = Flask(__name__)

@app.route('/api/invoice_detector', methods=['POST'])
def predict():
    data = jsonify(request.get_json())
    base64_encoded_str = data['image']
    image = convert_from_base64(base64_encoded_str)

    gray, rotated, binary, cropped = process_image(image)

    gray_encoded_str = convert_to_base64(gray)
    rotated_encoded_str = convert_to_base64(rotated)
    binary_encoded_str = convert_to_base64(binary)
    cropped_encoded_str = convert_to_base64(cropped)

    invoice_information = extract_string_from_image(cropped)
    profile_info, order_details, order_summary = process_output(invoice_information)
    
    return {
        'original': image,
        'gray': gray_encoded_str,
        'rotated': rotated_encoded_str,
        'binary': binary_encoded_str,
        'cropped': cropped_encoded_str,
        'invoice_information': invoice_information,
        'profile': profile_info,
        'order_details': order_details,
        'order_summary': order_summary,
    }
