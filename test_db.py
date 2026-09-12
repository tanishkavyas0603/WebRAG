import sqlite3
conn = sqlite3.connect('webrag.db')
c = conn.cursor()
c.execute("SELECT content FROM messages WHERE content LIKE '%Failed to generate answer%'")
rows = c.fetchall()
print(rows)
