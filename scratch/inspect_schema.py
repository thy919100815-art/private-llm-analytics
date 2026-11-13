import sqlite3

conn = sqlite3.connect("results/privacy_state.db")
cursor = conn.cursor()
for table in ["privacy_epochs", "lifetime_privacy_exposure", "privacy_ledger", "release_store"]:
    cursor.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cursor.fetchall()]
    print(f"{table} columns: {cols}")
