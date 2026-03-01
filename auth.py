import hashlib
from datetime import date
import os
from dotenv import load_dotenv

load_dotenv()

# Salt ensures the hash is unique to your app
SECRET_SALT = os.getenv("AUTH_SECRET_SALT", "default_salt_change_me")

def get_daily_password():
    """Generates a deterministic 6-character code based on the date."""
    today = str(date.today())
    combined = f"{SECRET_SALT}-{today}"
    hash_object = hashlib.sha256(combined.encode())
    # We take a 6-char slice and make it uppercase for a 'token' feel
    return hash_object.hexdigest()[:6].upper()

def verify_password(input_pw):
    """Verifies the guest password."""
    return input_pw.upper() == get_daily_password()
