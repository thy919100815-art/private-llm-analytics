with open("src/private_llm_analytics/web/static/app.js", "r", encoding="utf-8") as f:
    text = f.read()

import re
funcs = re.findall(r"function ([a-zA-Z0-9_]+)\(", text)
print(f"Total functions in app.js: {len(funcs)}")
print("Sample function names:", funcs[:30])

# Check how dataset status and privacy status are updated in DOM
print("\nMentions of updatePrivacyStatus or similar:")
for line in text.splitlines():
    if "function update" in line or "function render" in line or "function load" in line:
        print(" ", line.strip()[:80])
