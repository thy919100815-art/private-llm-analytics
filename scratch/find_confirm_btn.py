with open("src/private_llm_analytics/web/static/app.js", "r", encoding="utf-8") as f:
    text = f.read()

import re
matches = [m.start() for m in re.finditer(r"btn-confirm-release", text)]
print("btn-confirm-release matches at character positions:", matches)
for m in matches:
    start = max(0, m - 50)
    end = min(len(text), m + 250)
    print("--- Snippet ---")
    print(text[start:end])
