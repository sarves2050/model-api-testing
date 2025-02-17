from fastapi import APIRouter, HTTPException
from db import contact_collection
from schema.contactMongo import ContactUsSchema

router = APIRouter()

@router.post("/contact")
async def submit_contact_form(contact: ContactUsSchema):
    try:
        contact_dict = contact.dict()
        result = await contact_collection.insert_one(contact_dict)

        if result.inserted_id:
            return {"message": "Your message has been received. We'll get back to you soon!"}
        else:
            raise HTTPException(status_code=500, detail="Failed to save the contact request.")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
