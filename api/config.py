from pytesseract import pytesseract

# set the path to Tesseract executable
pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

SERVER_URL = "http://127.0.0.1:8001/info"
