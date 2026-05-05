import sqlite3

conn = sqlite3.connect("u2u_demo.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    drone TEXT,
    action TEXT,
    x REAL,
    y REAL,
    z REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

def log(drone, action, x=0, y=0, z=0):
    cursor.execute(
        "INSERT INTO logs (drone, action, x, y, z) VALUES (?, ?, ?, ?, ?)",
        (drone, action, x, y, z)
    )
    conn.commit()