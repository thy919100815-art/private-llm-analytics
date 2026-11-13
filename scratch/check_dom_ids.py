import re

with open('src/private_llm_analytics/web/templates/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

with open('src/private_llm_analytics/web/static/app.js', 'r', encoding='utf-8') as f:
    js = f.read()

html_ids = set(re.findall(r'id=["\']([a-zA-Z0-9_\-]+)["\']', html))
print(f'Total HTML IDs: {len(html_ids)}')

js_get_ids = set(re.findall(r'getElementById\(["\']([a-zA-Z0-9_\-]+)["\']\)', js))
print(f'Total JS getElementById IDs: {len(js_get_ids)}')

missing = js_get_ids - html_ids
print(f'Missing IDs ({len(missing)}):')
for m in sorted(missing):
    print('  -', m)
