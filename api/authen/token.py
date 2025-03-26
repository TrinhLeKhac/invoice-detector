from typing import Union
from datetime import timedelta, datetime, timezone
import jwt
from config import settings

def verify_password(username: str, password: str) -> bool:
    """
    Xác thực username và password.
    """
    return username == settings.USERNAME_AUTH and password == settings.PASSWORD_AUTH

def generate_token(username: Union[str, Any]) -> str:
    """
    Tạo JWT token với thông tin username.
    """
    expire_time = datetime.now(timezone.utc) + timedelta(seconds=settings.ACCESS_TOKEN_EXPIRE_SECONDS)
    to_encode = {
        "exp": expire_time.timestamp(),  
        "username": username  
    }
    
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.SECURITY_ALGORITHM)