from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timezone, timedelta
import jwt
from config import SECRET_KEY, SECURITY_ALGORITHM, ACCESS_TOKEN_EXPIRE_SECONDS, USERNAME_AUTH, PASSWORD_AUTH

# Security scheme
oauth2_scheme = HTTPBearer(scheme_name="Authorization")


def verify_password(username: str, password: str) -> bool:
    """
    Verify the username and password against the configuration values.
    """
    return username == USERNAME_AUTH and password == PASSWORD_AUTH


def generate_token(username: str) -> str:
    """
    Generate a JWT token with the username and expiration time.
    """
    expire_time = datetime.now(timezone.utc) + timedelta(seconds=ACCESS_TOKEN_EXPIRE_SECONDS)
    payload = {"exp": expire_time.timestamp(), "username": username}
    
    return jwt.encode(payload, SECRET_KEY, algorithm=SECURITY_ALGORITHM)


def validate_token(credentials: HTTPAuthorizationCredentials = Depends(oauth2_scheme)) -> str:
    """
    Validate the JWT token and return the username if it is valid.
    """
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[SECURITY_ALGORITHM])

        if payload.get("exp", 0) < datetime.now(timezone.utc).timestamp():
            raise HTTPException(status_code=403, detail="Token expired")

        return payload.get("username", "")

    except jwt.PyJWTError:
        raise HTTPException(status_code=403, detail="Invalid token")
