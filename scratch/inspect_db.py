import sqlite3

conn = sqlite3.connect("results/privacy_state.db")
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cursor.fetchall()]
print("Tables:", tables)

for t in tables:
    cursor.execute(f"SELECT COUNT(*) FROM {t}")
    cnt = cursor.fetchone()[0]
    print(f"  {t}: {cnt} rows")
