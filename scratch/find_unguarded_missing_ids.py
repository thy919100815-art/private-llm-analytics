import re

with open('src/private_llm_analytics/web/templates/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

with open('src/private_llm_analytics/web/static/app.js', 'r', encoding='utf-8') as f:
    js_lines = f.readlines()

html_ids = set(re.findall(r'id=["\']([a-zA-Z0-9_\-]+)["\']', html))

for i, line in enumerate(js_lines, 1):
    ids_in_line = re.findall(r'getElementById\(["\']([a-zA-Z0-9_\-]+)["\']\)', line)
    for id_val in ids_in_line:
        if id_val not in html_ids:
            print(f'Line {i}: {id_val} not in html! Line content: {line.strip()[:100]}')
