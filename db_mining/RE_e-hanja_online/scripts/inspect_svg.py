import re

with open(r'd:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3/db_mining/RE_e-hanja_online/samples/9451.svg', 'r', encoding='utf-8') as f:
    s = f.read()

paths = re.findall(r'<path\b[^>]*></path>', s)
print('total paths:', len(paths))

d_paths = [p for p in paths if re.search(r'id="U9451d\d+"', p)]
c_paths = [p for p in paths if re.search(r'id="U9451c\d+"', p)]
print('d paths (outline):', len(d_paths))
print('c paths (centerline):', len(c_paths))

def stroke_num(p, prefix):
    m = re.search(fr'id="U9451{prefix}(\d+)"', p)
    return int(m.group(1)) if m else -1

d_paths.sort(key=lambda p: stroke_num(p, 'd'))
c_paths.sort(key=lambda p: stroke_num(p, 'c'))

print()
print('>>> d1 (outline of stroke 1):')
print(d_paths[0][:360])
print()
print('>>> c1 (centerline of stroke 1):')
print(c_paths[0][:360])
print()

radicals_d = [p for p in d_paths if 'stroke-radical' in p]
radicals_c = [p for p in c_paths if 'stroke-radical' in p]
print(f'radical-labeled: d={len(radicals_d)} c={len(radicals_c)}')
print('  radical d strokes:', [stroke_num(p, 'd') for p in radicals_d])
print('  radical c strokes:', [stroke_num(p, 'c') for p in radicals_c])
