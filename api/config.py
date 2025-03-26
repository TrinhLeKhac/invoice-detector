from pytesseract import pytesseract

# set the path to Tesseract executable
pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

SERVER_URL = "http://127.0.0.1:8001/info"
API_CHECKED = False
SECRET_KEY="123456"
USERNAME_AUTH="abc"
PASSWORD_AUTH="123"
SECURITY_ALGORITHM = 'HS256'
ACCESS_TOKEN_EXPIRE_SECONDS = 2*24*60*60