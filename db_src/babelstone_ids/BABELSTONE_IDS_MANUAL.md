# BabelStone IDS Manual

이 문서는 [babelstone_ids](./babelstone_ids) 디렉터리의 구조와 활용법을
정리한다. 원천은 Andrew West (魏安) 가 유지하던 [BabelStone IDS database](https://www.babelstone.co.uk/CJK/IDS.HTML)
의 IDS.TXT 파일 (Unicode 16.0, 2025-06-27 release).

## Source And Licensing

- **프로젝트**: BabelStone CJK Database
- **저자**: Andrew West (1960-2025)
- **원본 URL**: `https://www.babelstone.co.uk/CJK/IDS.TXT`
- **수집일**: 2026-04-23
- **Unicode 기준**: **16.0 (2024-09)**
- **라이선스**: 원 파일에 명시 없음 — Unicode L2/21-161 에서 copyright
  우려가 제기된 바 있다. 프로젝트 **내부 학습용으로 사용**, 외부 재배포
  시 원저자 상속인 / Unicode Consortium 의견 확인 필요.

### License Caveat

Andrew West 가 2025 년 7 월 별세해 upstream 유지보수가 중단되었다. GitHub
mirror (`qundao/mirror-babelstone-ids`, `mandel59/babelstone-ids`) 로 데이터는
보존되어 있다. 이 프로젝트에서는 원본 snapshot 을 db_mining 에 보존하고,
정제본만 db_src 에 둔다.

## File Inventory

```
db_mining/RE_babelstone_ids/
  data/IDS.TXT          # 원본 (3.13 MB, 97,680 entries, UTF-8 BOM + CRLF)
  process.py            # 파서 → db_src 변환 스크립트

db_src/babelstone_ids/
  ids.jsonl             # 정제본 (multi-alternate 보존)
  ids_primary.jsonl     # primary IDS 만 (간단 lookup 용)
  stats.json            # 블록별 통계
```

## 원본 포맷

BabelStone 의 한 줄 포맷:

```
U+4E00<TAB>一<TAB>^一$(GHTJKPV)
U+4E06<TAB>丆<TAB>^⿱一丿$(GK)<TAB>^⿱一㇒$(X)
U+9451<TAB>鑑<TAB>^⿰釒監$(GHTJKP)
```

- `^<IDS>$(<flags>)` 로 IDS 를 감싼다. `^` 와 `$` 는 regex-style anchor,
  괄호 안은 region flag.
- **Region flag** (Unicode IRG 분류):
  - `G` — PRC (중국)
  - `H` — Hong Kong
  - `T` — Taiwan
  - `J` — Japan (JIS)
  - `K` — South Korea
  - `P` — DPRK (North Korea)
  - `V` — Vietnam
  - `X` — extended / minor (사용 빈도 낮은 variant)
- 한 줄에 여러 탭으로 **region-specific alternates** 병기. 예: 丆 는 G+K
  에서 `⿱一丿`, X 에서 `⿱一㇒` 로 형태 다름.

## 정제본 Schema

### `ids.jsonl`

한 줄 = 한 codepoint. Multi-alternate 보존.

```json
{
  "codepoint": "U+9451",
  "char": "鑑",
  "ids": ["⿰釒監"],
  "regions": [["G", "H", "J", "K", "P", "T"]]
}
```

- `ids` 와 `regions` 는 같은 길이 list. index `i` 의 IDS 가 `regions[i]`
  region 에서 쓰인다는 뜻.
- multi-alternate 예:
  ```json
  {
    "codepoint": "U+4E06",
    "char": "丆",
    "ids": ["⿱一丿", "⿱一㇒"],
    "regions": [["G", "K"], ["X"]]
  }
  ```

### `ids_primary.jsonl`

간단 lookup 용. 첫 번째 (primary) IDS 만 수록.

```json
{"codepoint": "U+9451", "char": "鑑", "ids": "⿰釒監"}
```

## 통계 (2026-04-23 local)

- **총 엔트리**: 97,649
- **unique codepoints**: 97,649 (중복 없음)
- **multi-alternate (2개 이상)**: 9,525 (9.8%)

블록별 분포:

| Block | Entries |
|---|---:|
| CJK Ext B (SMP) | 42,718 |
| CJK Unified | 20,965 |
| CJK Ext F | 7,473 |
| CJK Ext A | 6,590 |
| CJK Ext E | 5,762 |
| CJK Ext G | 4,939 |
| CJK Ext H | 4,192 |
| CJK Ext C | 4,154 |
| CJK Ext I | 622 |
| CJK Ext D | 222 |
| CJK Compat | 12 |

## 핵심 가치 / 한계

### 가치
1. **97,649 codepoint 전체에 region 주석 포함 IDS 제공** — 한국식 자형
   (K flag) 과 중국식 자형 (G flag) 이 다를 때 region-specific IDS 를
   고를 수 있음.
2. **Radical-position 세분화** — 金 부수가 왼쪽에 올 때 `釒` (radical
   form) 으로 기술. CHISE / cjkvi 는 canonical `金` 만 씀.
3. **Ext G/H/I (최신 Unicode 확장)** 커버리지 완전.

### 한계
1. 저자 별세로 **유지보수 중단** — Unicode 17.0 (2025-09) 부터 새로 추가
   된 Ext J (4,298 자) 는 포함되지 않음. CHISE 가 해당 영역 cover.
2. 라이선스 명시 없음 — 외부 배포 시 법률 검토 필요.
3. `釒` 같은 radical-position 변형형이 IDS 에 섞여 있어 일부 downstream
   tool (canonical 부품 매칭) 이 추가 normalize 필요.

## 재생성 명령

원본 다운로드 + 정제:

```bash
# 원본 (이미 수집됨, 재수집 시에만)
curl -sSL -A "Mozilla/5.0" \
  -o db_mining/RE_babelstone_ids/data/IDS.TXT \
  https://www.babelstone.co.uk/CJK/IDS.TXT

# 정제
python db_mining/RE_babelstone_ids/process.py
```

## 관련 문서

- [doc/17_CANONICAL_V3_PLAN.md](../doc/17_CANONICAL_V3_PLAN.md) — canonical_v3
  schema 설계에서 IDS tree 정보의 통합 전략
- [CHISE_IDS_MANUAL.md](./CHISE_IDS_MANUAL.md) — 자매 소스
- [CJKVI_IDS_MANUAL.md](./CJKVI_IDS_MANUAL.md) — 자매 소스
- [L2/21-118 kIDS preliminary proposal](https://www.unicode.org/L2/L2021/21118-kids-preliminary.pdf)
- [L2/21-161 IDS copyright concerns](https://www.unicode.org/L2/L2021/21161-ids-copyright.pdf)
