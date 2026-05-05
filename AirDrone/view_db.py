import sqlite3

conn = sqlite3.connect("u2u_demo.db")
cursor = conn.cursor()

rows = cursor.execute("SELECT * FROM logs").fetchall()

for row in rows:
    print(row)