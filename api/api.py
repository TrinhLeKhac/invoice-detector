from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from utils.image.base64 import convert_from_base64, convert_to_base64
from utils.image.image_processing import (processing_image,
                                          processing_image_full)
from utils.image.ocr_parser import parse_general_information
from utils.text.handler import handle_general_information

app = FastAPI()


class InvoiceRequest(BaseModel):
    image: str


@app.post("/api/invoice_detector_demo")
async def predict(data: InvoiceRequest):
    try:
        image_encoded_str = data.image
        image = convert_from_base64(image_encoded_str)

        cropped, deskewed, table_roi, table_information = processing_image_full(image)

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/invoice_detector")
async def predict(data: InvoiceRequest):
    try:
        image_encoded_str = data.image
        image = convert_from_base64(image_encoded_str)

        cropped = processing_image(image)

        general_information = parse_general_information(cropped)
        profile_info, order_summary = handle_general_information(general_information)

        return {
            "name": profile_info["customer_name"],
            "phone": profile_info["customer_phone"],
            "address": profile_info["address"],
            "commune": profile_info["commune"],
            "district": profile_info["district"],
            "province": profile_info["province"],
            "total_quantity": order_summary["total_quantity"],
            "total_amount": order_summary["total_amount"],
            "discount": order_summary["discount"],
            "monetary": order_summary["monetary"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
