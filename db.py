import motor.motor_asyncio
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB connection string from .env
MONGO_URI = os.getenv("MONGODB_URL","mongodb+srv://nanu14581:nanu123@cluster0.xwe83.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0" )
print(MONGO_URI)
# Connect to MongoDB
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client["BitBeeAI"]  # Database name
users_collection = db["users"]  
chats_collection = db["chat_history"] 
storeData_collection = db["store_data"] 
contact_collection = db["contact_us"]

# Check MongoDB connection
async def check_database_connection():
    try:
        await client.server_info()  # Test connection
        print(" Data is connected")
    except Exception as e:
        print(" Database connection failed:", e)
