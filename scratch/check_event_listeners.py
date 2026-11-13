import re

with open('src/private_llm_analytics/web/templates/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

with open('src/private_llm_analytics/web/static/app.js', 'r', encoding='utf-8') as f:
    js = f.read()

matches = re.findall(r'document\.getElementById\(["\']([a-zA-Z0-9_\-]+)["\']\)\.addEventListener', js)
for m in matches:
    if f'id="{m}"' not in html and f"id='{m}'" not in html:
        print(f'MISSING in HTML: {m}')
