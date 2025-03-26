from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import httpx
import logging
import jwt
from datetime import datetime, timezone

from config import SERVER_URL, API_CHECKED
from utils.image.base64 import convert_from_base64, convert_to_base64
from utils.image.image_processing import processing_image, processing_image_full
from utils.image.ocr_parser import parse_general_information
from utils.text.handler import handle_general_information
from authen import verify_password, generate_token, validate_token


# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define request and response schemas
class LoginModel(BaseModel):
    username: str 
    password: str 

class InvoiceRequestDemo(BaseModel):
    image: str

class InvoiceRequest(BaseModel):
    image: str
    shop_code: str

class PhoneRequest(BaseModel):
    phone: str

class PhoneResponse(BaseModel):
    name: str 
    address: str 


@app.post("/login")
def login(request_data: LoginModel):
    """
    Xác thực người dùng và trả về token nếu thành công.
    """
    if not verify_password(request_data.username, request_data.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    return {
        "error": False,
        "message": "Login successful",
        "data": {"token": generate_token(request_data.username)}
    }


@app.post("/api/info", response_model=PhoneResponse, dependencies=[Depends(validate_token)])
async def fetch_user_info(request_data: PhoneRequest):
    """
    Call API to retrieve user information based on phone number
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(SERVER_URL, json={"phone": request_data.phone})
        response.raise_for_status()
        return PhoneResponse(**response.json())
    except httpx.HTTPStatusError as e:
        status_code = e.response.status_code
        detail = "Phone number not found" if status_code == 404 else "Error fetching data"
        raise HTTPException(status_code=status_code, detail=detail)
    except Exception as e:
        logger.error(f"Error fetching user info: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")



@app.post("/api/invoice_detector_demo")
async def predict_demo(data: InvoiceRequestDemo):
    """
    Process invoice image demo, extract information from the image, and return the data.
    """
    try:
        image = convert_from_base64(data.image)
        cropped, deskewed, table_roi, table_information = processing_image_full(image)

        general_information = parse_general_information(cropped)
        profile_info, order_summary = handle_general_information(general_information)
        
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
async def predict_invoice(data: InvoiceRequest):
    """
    Process invoice image and verify customer information with server data.
    """
    try:
        image = convert_from_base64(data.image)
        cropped = processing_image(image)

        general_information = parse_general_information(cropped)
        profile_info, order_summary = handle_general_information(general_information)
        
        # Verify customer information

        phone_checked = 0
        name_checked = 0
        address_checked = 0

        if API_CHECKED:
            user_info = None
            try:
                user_info = await fetch_user_info(PhoneRequest(phone=profile_info["customer_phone"]))
            except HTTPException:
                pass

            phone_checked = int(user_info is not None)
            name_checked = int(user_info is not None and user_info.name == profile_info.get("customer_name", ""))
            address_checked = int(user_info is not None and user_info.address == profile_info.get("address", ""))


        return {
            "shop_code": data.shop_code,
            "name": profile_info.get("customer_name", ""),
            "phone": profile_info.get("customer_phone", ""),
            "address": profile_info.get("address", ""),
            "commune": profile_info.get("commune", ""),
            "district": profile_info.get("district", ""),
            "province": profile_info.get("province", ""),
            "phone_checked": phone_checked,
            "name_checked": name_checked,
            "address_checked": address_checked,
            "total_quantity": order_summary.get("total_quantity", 0),
            "total_amount": order_summary.get("total_amount", 0),
            "discount": order_summary.get("discount", 0),
            "monetary": order_summary.get("monetary", 0),
        }
    except Exception as e:
        logger.error(f"Error in predict_invoice: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")



