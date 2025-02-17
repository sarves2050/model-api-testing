from pydantic import BaseModel, EmailStr

class ContactUsSchema(BaseModel):
    name: str
    email: EmailStr
    subject: str
    message: str
