import re

with open("src/private_llm_analytics/web/templates/index.html", "r", encoding="utf-8") as f:
    text = f.read()

ids = re.findall(r'id=["\']([^"\']+)["\']', text)
print(f"Total HTML elements with IDs: {len(ids)}")
print("Sample IDs:", ids[:50])

# Let's see section/div class names
sections = re.findall(r'<(section|div)[^>]*class=["\']([^"\']+)["\']', text)
print("\nUnique classes:", len(set([c for _, c in sections])))
