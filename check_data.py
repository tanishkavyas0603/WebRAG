import sqlite3

conn = sqlite3.connect('webrag.db')
cursor = conn.cursor()

# Get users
cursor.execute('SELECT id, email FROM users')
users = cursor.fetchall()
print('Users in database:')
for user in users:
    print(f'  ID: {user[0]}, Email: {user[1]}')

# Get conversations
cursor.execute('SELECT id, user_id, document_id, title FROM conversations')
conversations = cursor.fetchall()
print('\nConversations in database:')
for conv in conversations:
    print(f'  ID: {conv[0]}, User: {conv[1]}, Doc: {conv[2]}, Title: {conv[3]}')

# Get documents
cursor.execute('SELECT id, user_id, title, status FROM documents')
docs = cursor.fetchall()
print('\nDocuments in database:')
for doc in docs:
    print(f'  ID: {doc[0]}, User: {doc[1]}, Title: {doc[2]}, Status: {doc[3]}')
    
# Specifically check conversation 14
print('\n--- CONVERSATION 14 DETAILS ---')
cursor.execute('SELECT id, user_id, document_id, title FROM conversations WHERE id = 14')
conv14 = cursor.fetchone()
if conv14:
    print(f'ID: {conv14[0]}, User: {conv14[1]}, Doc: {conv14[2]}, Title: {conv14[3]}')
else:
    print('Conversation 14 not found')

conn.close()
