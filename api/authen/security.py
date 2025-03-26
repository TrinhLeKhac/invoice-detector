from datetime import datetime, timezone
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer
from pydantic import ValidationError
from config import settings

reusable_oauth2 = HTTPBearer(scheme_name='Authorization')

def validate_token(http_authorization_credentials=Depends(reusable_oauth2)) -> str:
    """
    Decode JWT token để lấy username.
    """
    try:
        payload = jwt.decode(
            http_authorization_credentials.credentials, 
            settings.SECRET_KEY, 
            algorithms=[settings.SECURITY_ALGORITHM]
        )

        exp_timestamp = payload.get("exp")
        if exp_timestamp is None or exp_timestamp < datetime.now(timezone.utc).timestamp():
            raise HTTPException(status_code=403, detail="Token expired")

        return payload.get("username")

    except (jwt.PyJWTError, ValidationError):
        raise HTTPException(
            status_code=403,
            detail={"error": True, "message": "Invalid token", "data": []},
        )