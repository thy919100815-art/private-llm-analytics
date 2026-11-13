import sqlite3
import json

conn = sqlite3.connect("results/privacy_state.db")
cursor = conn.cursor()

cursor.execute("SELECT epoch_id, status, effective_epsilon, epsilon_cap, workload_spent_json, reserve_spent FROM privacy_epochs")
for row in cursor.fetchall():
    print("privacy_epoch:", row[0], row[1], row[2], row[3], row[4], row[5])

cursor.execute("SELECT * FROM lifetime_privacy_exposure")
for row in cursor.fetchall():
    print("lifetime_exposure:", row)
