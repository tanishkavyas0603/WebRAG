import requests
import json

BASE_URL = "http://localhost:8000"

# Step 1: Register a test user (or it might already exist)
print("=== Step 1: Register/Login ===")
register_data = {
    "email": "test@example.com",
    "password": "TestPass123!"
}

# Try to register
resp = requests.post(f"{BASE_URL}/api/auth/register", json=register_data)
print(f"Register status: {resp.status_code}")
if resp.status_code != 200:
    print(f"Register response: {resp.json()}")

# Login
login_data = {
    "username": "test@example.com",
    "password": "TestPass123!"
}
resp = requests.post(f"{BASE_URL}/api/auth/login", data=login_data)
print(f"Login status: {resp.status_code}")
if resp.status_code != 200:
    print(f"Login response: {resp.json()}")
    exit(1)

token = resp.json()["access_token"]
print(f"Got token: {token[:30]}...")

headers = {"Authorization": f"Bearer {token}"}

# Step 2: Get conversations
print("\n=== Step 2: Get Conversations ===")
resp = requests.get(f"{BASE_URL}/api/conversations", headers=headers)
print(f"Conversations status: {resp.status_code}")
conversations = resp.json()
print(f"Found {len(conversations)} conversations")
for conv in conversations[:3]:
    print(f"  Conversation {conv['id']}: {conv.get('title', 'N/A')}")

# Step 3: Send a message to conversation with document that's ready
# Let's find a conversation first
if conversations:
    conv_id = conversations[0]["id"]
    print(f"\n=== Step 3: Send Message to Conversation {conv_id} ===")
    
    message_data = {"message": "What is HTTP?"}
    resp = requests.post(f"{BASE_URL}/api/conversations/{conv_id}/messages", json=message_data, headers=headers)
    print(f"Send message status: {resp.status_code}")
    print(f"Response:")
    print(json.dumps(resp.json(), indent=2))
else:
    print("No conversations found. Cannot test message sending.")
