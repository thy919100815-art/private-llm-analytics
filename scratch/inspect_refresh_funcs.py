with open("src/private_llm_analytics/web/static/app.js", "r", encoding="utf-8") as f:
    lines = f.readlines()

def print_func(name):
    start = -1
    for i, line in enumerate(lines):
        if f"function {name}" in line or f"async function {name}" in line:
            start = i
            break
    if start >= 0:
        print(f"--- Function {name} (lines {start+1} - {start+60}) ---")
        print(''.join(lines[start:start+60]))

print_func("refreshDatasetStatus")
print_func("refreshPrivacyStatus")
