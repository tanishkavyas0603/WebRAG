import os
import requests
from app.core.database import SessionLocal
from app.models.db import User, Conversation
from app.core.security import create_access_token

def main():
    db = SessionLocal()
    user = db.query(User).first()
    if not user:
        print("No user found")
        return
        
    print(f"Found user: {user.email}")
    token = create_access_token({"sub": str(user.id)})
    
    conv = db.query(Conversation).filter(Conversation.user_id == user.id).first()
    if not conv:
        print("No conversation found for user")
        return
        
    print(f"Using conversation {conv.id}")
    
    url = f"http://localhost:8000/api/conversations/{conv.id}/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    print("Sending request...")
    res = requests.post(url, json={"message": "What is the summary?"}, headers=headers)
    print("Status:", res.status_code)
    print("Response JSON:", res.json())

if __name__ == "__main__":
    main()
