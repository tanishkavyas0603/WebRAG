import requests
import json
import time

BASE_URL = "http://localhost:8000"

# Step 1: Register a new test user
print("=== Step 1: Register New User ===")
test_email = f"debug_test_{int(time.time())}@example.com"
test_password = "TestPass123!"

register_data = {
    "email": test_email,
    "password": test_password
}

resp = requests.post(f"{BASE_URL}/api/auth/register", json=register_data)
print(f"Register status: {resp.status_code}")
if resp.status_code != 200:
    print(f"Register response: {resp.json()}")
    exit(1)

print(f"Registered user: {test_email}")

# Step 2: Login
print("\n=== Step 2: Login ===")
login_data = {
    "username": test_email,
    "password": test_password
}
resp = requests.post(f"{BASE_URL}/api/auth/login", data=login_data)
print(f"Login status: {resp.status_code}")
if resp.status_code != 200:
    print(f"Login response: {resp.json()}")
    exit(1)

token = resp.json()["access_token"]
print(f"Got token: {token[:30]}...")

headers = {"Authorization": f"Bearer {token}"}

# Step 3: Ingest a document
print("\n=== Step 3: Ingest Document ===")
ingest_data = {
    "url": "https://httpwg.org/specs/rfc7230.html"  # Simple HTTP spec page
}
resp = requests.post(f"{BASE_URL}/api/documents/ingest", json=ingest_data, headers=headers)
print(f"Ingest status: {resp.status_code}")
if resp.status_code != 200:
    print(f"Ingest response: {resp.json()}")
    exit(1)

doc_id = resp.json()["id"]
print(f"Document ingested with ID: {doc_id}")

# Wait for document processing
print("\n=== Step 4: Wait for Document Processing ===")
for i in range(30):
    resp = requests.get(f"{BASE_URL}/api/documents/{doc_id}/status", headers=headers)
    doc_status = resp.json()["status"]
    print(f"Document status: {doc_status}")
    if doc_status == "ready":
        break
    time.sleep(1)

if doc_status != "ready":
    print(f"Document did not become ready. Final status: {doc_status}")
    exit(1)

# Step 5: Create conversation
print("\n=== Step 5: Create Conversation ===")
conv_data = {"document_id": doc_id}
resp = requests.post(f"{BASE_URL}/api/conversations", json=conv_data, headers=headers)
print(f"Create conversation status: {resp.status_code}")
if resp.status_code != 200:
    print(f"Create conversation response: {resp.json()}")
    exit(1)

conv_id = resp.json()["id"]
print(f"Conversation created with ID: {conv_id}")

# Step 6: Send message
print(f"\n=== Step 6: Send Message to Conversation {conv_id} ===")
message_data = {"message": "What is HTTP?"}
resp = requests.post(f"{BASE_URL}/api/conversations/{conv_id}/messages", json=message_data, headers=headers)
print(f"Send message status: {resp.status_code}")
print(f"Response:")
print(json.dumps(resp.json(), indent=2))

if resp.status_code != 200:
    print("\n!!! ERROR !!!")
    print(f"Failed with status {resp.status_code}")
    if "detail" in resp.json():
        print(f"Error detail: {resp.json()['detail']}")
