import glob, re

for f in glob.glob('tests/**/*.py', recursive=True):
    with open(f, 'r', encoding='utf-8') as fp:
        content = fp.read()
    if 'html' in content:
        matches = re.findall(r'assert\s+[\'"][^\n]+?[\'"]\s+in\s+html', content)
        not_matches = re.findall(r'assert\s+[\'"][^\n]+?[\'"]\s+not\s+in\s+html', content)
        if matches or not_matches:
            print(f'=== {f} ===')
            for m in matches:
                print('   IN:', m)
            for m in not_matches:
                print('  NOT:', m)
