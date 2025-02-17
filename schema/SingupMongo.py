from pydantic import BaseModel, EmailStr
import random

class SignupRequest(BaseModel):
    fullname: str
 
    email: EmailStr
    password: str
    terms_agreed: bool 

class OTPVerifyRequest(BaseModel):
    email: EmailStr
    otp: str

def generate_otp():
    """Generate a 6-digit OTP."""
    return str(random.randint(100000, 999999))
