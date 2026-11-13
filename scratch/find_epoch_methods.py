with open("src/private_llm_analytics/privacy/state.py", "r", encoding="utf-8") as f:
    text = f.read()

import re
matches = re.findall(r"def (.*?epoch.*?|.*?lifetime.*?)\(", text)
print("Epoch/Lifetime methods in state.py:")
for m in matches:
    print(" ", m)
