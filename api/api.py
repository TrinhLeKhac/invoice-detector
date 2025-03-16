from flask import Flask, jsonify, request

from utils.image.base64 import convert_from_base64, convert_to_base64
from utils.image.image_processing import processing_image
from utils.image.ocr_parser import parse_general_information
from utils.text.handler import handle_general_information

app = Flask(__name__)


@app.route("/api/invoice_detector", methods=["POST"])
def predict():

    data = request.get_json()
    image_encoded_str = data["image"]
    # print(base64_encoded_str)

    image = convert_from_base64(image_encoded_str)
    print(image.shape)

    cropped, deskewed, table_roi, table_information = processing_image(image)

    p1_image_encoded_str = convert_to_base64(cropped)
    p2_image_encoded_str = convert_to_base64(deskewed)
    p3_image_encoded_str = convert_to_base64(table_roi)

    general_information = parse_general_information(cropped)
    profile_info, order_summary = handle_general_information(general_information)

    return {
        "original": image_encoded_str,
        "gray": p1_image_encoded_str,
        "binary": p2_image_encoded_str,
        "cropped": p3_image_encoded_str,
        "invoice_information": general_information,
        "profile": profile_info,
        "order_details": table_information,
        "order_summary": order_summary,
    }
