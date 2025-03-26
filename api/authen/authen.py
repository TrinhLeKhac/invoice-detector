from fastapi import APIRouter, HTTPException
from scripts.api.auth.token import verify_password, generate_token
from scripts.api.schemas import LoginModel

router = APIRouter()

@router.post("")
def login(request_data: LoginModel):
    """
    Xác thực người dùng và trả về token nếu thành công.
    """
    print(f'[x] request_data: {request_data.__dict__}')
    
    if not verify_password(username=request_data.username, password=request_data.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    token = generate_token(request_data.username)
    
    return {
        "error": False,
        "message": "Login successful",
        "data": {
            "token": token
        }
    }