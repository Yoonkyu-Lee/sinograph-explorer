import re, sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
with open(r'd:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3/db_mining/RE_e-hanja_online/samples/detailA_鑑.html', 'r', encoding='utf-8') as f:
    html = f.read()

badges = [
    ('훈&nbsp;&nbsp;음', '훈음'),
    ('자&nbsp;&nbsp;해', '자해'),
    ('부&nbsp;&nbsp;수', '부수'),
    ('획&nbsp;&nbsp;수', '획수'),
    ('모양자', '모양자'),
    ('자&nbsp;&nbsp;원', '자원'),
    ('영&nbsp;&nbsp;문', '영문'),
    ('한어병음', '한어병음'),
    ('교육용', '교육용'),
    ('검정', '검정'),
    ('대법원', '대법원'),
    ('분&nbsp;&nbsp;류', '분류'),
    ('동자', '동자'),
    ('유의', '유의'),
]

for html_text, label in badges:
    pat = re.escape(html_text) + r'</span>(.*?)(?=<span class="w3-tag w3-round-large|$)'
    m = re.search(pat, html, re.DOTALL)
    if m:
        raw = m.group(1)
        text = re.sub(r'<script.*?</script>', '', raw, flags=re.DOTALL)
        text = re.sub(r'<style.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        print(f'[{label}]')
        print(f'  {text[:400]}')
        print()
