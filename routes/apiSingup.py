import os
import smtplib
import datetime
import jwt as pyjwt  # Ensure we are using the correct PyJWT library
from email.message import EmailMessage
from passlib.context import CryptContext
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException,Response
from db import users_collection
from schema.SingupMongo import SignupRequest, OTPVerifyRequest, generate_otp

# Load environment variables
load_dotenv()

# Password hashing setup
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

def hash_password(password: str) -> str:
    """Hash a password using Argon2."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)


import os
import smtplib
import datetime
from email.message import EmailMessage
from fastapi import HTTPException

def send_otp_email(email: str, otp: str):
    """Send OTP to user's email with a professional and well-designed format."""
    EMAIL_USER = os.getenv("EMAIL_USER")
    EMAIL_PASS = os.getenv("EMAIL_PASS")

    if not EMAIL_USER or not EMAIL_PASS:
        raise HTTPException(status_code=500, detail="Email credentials are missing in environment variables.")

    subject = "🔒 Secure Your BitBeeAI Account – One-Time Passcode Inside"

    html_content = f"""
    <html>
        <body style="font-family: Arial, sans-serif; background-color: #f9f9f9; padding: 40px;">
            <div style="max-width: 480px; margin: auto; background: #fff; padding: 30px; border-radius: 10px; 
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); border-left: 6px solid #4CAF50;">
                <h2 style="color: #333; text-align: center;">🔑 Your Secure Login Code</h2>
                <p style="font-size: 16px; color: #444; text-align: center;">
                    Use the following one-time passcode to verify your identity for BitBeeAI:
                </p>
                <div style="text-align: center; margin: 20px 0;">
                    <span style="display: inline-block; font-size: 26px; font-weight: bold; color: #ffffff; 
                                 background: #4CAF50; padding: 12px 24px; border-radius: 8px;">
                        {otp}
                    </span>
                </div>
                <p style="font-size: 14px; color: #666; text-align: center;">
                    This code is valid for <b>10 minutes</b>. Please do not share it with anyone.
                </p>
                <hr style="border: none; border-top: 1px solid #ddd; margin: 20px 0;">
                <p style="font-size: 13px; color: #888; text-align: center;">
                    If you didn't request this, you can safely ignore this email.<br>
                    Need help? Contact us at <a href="mailto:info@bitbeeai.com" style="color: #4CAF50;">info@bitbeeai.com</a>
                </p>
                <p style="text-align: center; font-size: 12px; color: #aaa;">
                    &copy; {datetime.datetime.now().year} BitBeeAI. All rights reserved.
                </p>
            </div>
        </body>
    </html>
    """

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"BitBeeAI <{EMAIL_USER}>"
    msg["To"] = email
    msg.set_content(f"Your OTP is: {otp}. It is valid for 10 minutes.")
    msg.add_alternative(html_content, subtype="html")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
        return True
    except Exception as e:
        print("❌ Email sending failed:", e)
        return False



# JWT Secret Key & Algorithm
JWT_SECRET = os.getenv("JWT_SECRET", "T2PmC8A8EU5Ke5PyUxy668x6FiXlp80jNDq7WCqIr-n3b5psMD3i0xs_4sAIfq8nubI5MIecHqhR0QG5em2_ww-BitModel-LEccOek9-tE_IJE7JVGkjs0MU_xYYEhFkCRY6Q-sGXS8UZdWspAKOeU_F50Whvmz-c3gq9SqdterKziR3Mm8A")  # Ensure this is set in your .env
JWT_ALGORITHM = "HS256"

def create_access_token(data: dict, expires_delta: datetime.timedelta = None):
    """Generate JWT Token."""
    to_encode = data.copy()
    if expires_delta:
        to_encode["exp"] = datetime.datetime.utcnow() + expires_delta
    else:
        to_encode["exp"] = datetime.datetime.utcnow() + datetime.timedelta(days=7)  # Default: 7 Days Validity
    
    return pyjwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)  # Use pyjwt.encode to avoid conflicts

router = APIRouter()

@router.post("/signup")
async def signup(user: SignupRequest):
    """Signup API with OTP Verification & Terms Agreement"""
    if not user.terms_agreed:
        raise HTTPException(status_code=400, detail="You must agree to the terms and conditions to proceed.")

    existing_user = await users_collection.find_one({"email": user.email})
    otp = generate_otp()
    hashed_password = hash_password(user.password)

    if existing_user:
        if existing_user["verified"]:
            raise HTTPException(status_code=400, detail="Email already registered.")
        else:
            await users_collection.update_one(
                {"email": user.email},
                {"$set": {"otp": otp, "otp_expiry": datetime.datetime.utcnow() + datetime.timedelta(minutes=10)}}
            )
    else:
        new_user = {
            "fullname": user.fullname,
            "email": user.email,
            "password": hashed_password,
            "otp": otp,
            "otp_expiry": datetime.datetime.utcnow() + datetime.timedelta(minutes=10),
            "verified": False,
            "terms_agreed": True,  # Store agreement in the database
            "token": None
        }
        await users_collection.insert_one(new_user)

    if send_otp_email(user.email, otp):
        return {"message": "OTP sent to email. Please verify to complete signup."}
    else:
        raise HTTPException(status_code=500, detail="Failed to send OTP.")


@router.post("/verify-otp")
async def verify_otp(data: OTPVerifyRequest, response: Response):
    """Verify OTP API and store JWT token for verified users"""
    user = await users_collection.find_one({"email": data.email})
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    if user["verified"]:
        raise HTTPException(status_code=400, detail="User already verified.")

    if datetime.datetime.utcnow() > user["otp_expiry"]:
        raise HTTPException(status_code=400, detail="OTP expired.")

    if user["otp"] != data.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP.")

    # Generate JWT Token
    token = create_access_token({"email": data.email})

    # Update user status and store token
    await users_collection.update_one(
        {"email": data.email},
        {"$set": {"verified": True, "otp": None, "otp_expiry": None, "token": token}}
    )

    # Store token in cookies
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=False,  # Prevent JavaScript access for security
        secure=True,  # Use only in HTTPS
        samesite="Lax"  # Adjust based on your authentication flow
    )

    return {"message": "Email verified successfully! Signup complete."}

