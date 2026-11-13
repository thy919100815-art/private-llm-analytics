with open("src/private_llm_analytics/web/static/app.js", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if "async function fetchConfig" in l:
        print("fetchConfig line:", i+1)
    if "function setupEventListeners" in l:
        print("setupEventListeners line:", i+1)
    if "async function executeRelease" in l:
        print("executeRelease line:", i+1)
