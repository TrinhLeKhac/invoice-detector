import logging
from datetime import datetime, timezone

import httpx
import jwt
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from authen import generate_token, validate_token, verify_password
from config import API_CHECKED, EXTERNAL_SERVER_URL
from utils.image.base64 import convert_from_base64, convert_to_base64
from utils.image.image_processing import (processing_image,
                                          processing_image_full)
from utils.image.ocr_parser import parse_general_information
from utils.text.handler import handle_general_information

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define request and response schemas
class LoginModel(BaseModel):
    username: str
    password: str


class ImageModel(BaseModel):
    image: str


class RequestImageModel(BaseModel):
    image: str
    shop_code: str


class RequestInfoModel(BaseModel):
    phone: str


class ResponseInfoModel(BaseModel):
    name: str
    address: str


@app.post("/login")
def login(request_data: LoginModel):
    """
    Authenticate the user and return a token if successful.
    """
    if not verify_password(request_data.username, request_data.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    return {
        "error": False,
        "message": "Login successful",
        "data": {"token": generate_token(request_data.username)},
    }


@app.post("/api/image", dependencies=[Depends(validate_token)])
async def fetch_image_info(request_data: RequestImageModel):
    """
    Call API to retrieve image information based on shop code
    """
    try:
        image_info = {"image": request_data.image, "shop_code": request_data.shop_code}
    except Exception as e:
        # logger.error(f"Error fetching image info: {str(e)}")
        # raise HTTPException(status_code=500, detail="Internal server error")
        return {"error": True, "message": f"Error fetching image: {str(e)}", "data": {}}
    return {
        "error": False,
        "message": "",
        "data": {
            "count": 1,
            "info": image_info,
        },
    }


async def fetching_user_info(request_data: RequestInfoModel) -> ResponseInfoModel:
    """
    Fetch user information from an external API based on phone number.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                EXTERNAL_SERVER_URL + "/api/info", json=request_data.dict()
            )
        response.raise_for_status()
        data = response.json()
        return ResponseInfoModel(**data)
    except Exception as e:
        logging.error(f"Unexpected error fetching user info: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/invoice_detector_demo")
async def predict_invoice_demo(request_data: ImageModel):
    """
    Process invoice image demo, extract information from the image, and return the data.
    """
    try:
        image = convert_from_base64(request_data.image)
        cropped, deskewed, table_roi, table_information = processing_image_full(image)

        general_information = parse_general_information(cropped)
        profile_info, order_summary = handle_general_information(general_information)
        print("COD: ", order_summary.get("cod", 0))
        return {
            "original": convert_to_base64(image),
            "gray": convert_to_base64(cropped),
            "binary": convert_to_base64(deskewed),
            "cropped": convert_to_base64(table_roi),
            "invoice_information": general_information,
            "profile": profile_info,
            "order_details": table_information,
            "order_summary": order_summary,
        }
    except Exception as e:
        logger.error(f"Error in predict_demo: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/invoice_detector", dependencies=[Depends(validate_token)])
async def predict_invoice(request_data: RequestImageModel):
    """
    Process invoice image and verify customer information with server data.
    """

    def check(info):
        info = re.sub(r"\s+", " ", info).strip()
        info = unidecode(info)
        return info.lower()

    try:
        image = convert_from_base64(request_data.image)
        cropped = processing_image(image)

        general_information = parse_general_information(cropped)
        profile_info, order_summary = handle_general_information(general_information)

        # Verify customer information

        phone_checked = 0
        name_checked = 0
        address_checked = 0

        if API_CHECKED & (profile_info["customer_phone"] != ""):
            user_info = None
            try:
                user_info = await fetching_user_info(
                    RequestInfoModel(phone=profile_info["customer_phone"])
                )
                user_info = user_info.dict()
            except Exception:
                pass

            if user_info is not None:
                phone_checked = 1
            if user_info is not None and (
                check(user_info["customer_name"])
                == check(profile_info["customer_name"])
            ):
                name_checked = 1
            if user_info is not None and (
                check(user_info["address"]) == check(profile_info["address"])
            ):
                address_checked = 1

        return {
            "shop_code": request_data.shop_code,
            "name": profile_info.get("customer_name", ""),
            "phone": profile_info.get("customer_phone", ""),
            "address": profile_info.get("address", ""),
            "commune": profile_info.get("commune", ""),
            "district": profile_info.get("district", ""),
            "province": profile_info.get("province", ""),
            "phone_checked": phone_checked,
            "name_checked": name_checked,
            "address_checked": address_checked,
            "cod_monetary": order_summary.get("cod", 0), 
            "total_quantity": order_summary.get("total_quantity", 0),
            "total_amount": order_summary.get("total_amount", 0),
            "discount": order_summary.get("discount", 0),
            "monetary": order_summary.get("monetary", 0),
        }
    except Exception as e:
        logger.error(f"Error in predict_invoice: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
