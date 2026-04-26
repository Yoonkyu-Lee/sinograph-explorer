# cjkvi-ids Manual

이 문서는 [cjkvi_ids](./cjkvi_ids) 디렉터리의 구조와 활용법을 정리한다.
원천은 [CJK VI Database](https://github.com/cjkvi/cjkvi-ids) 프로젝트의
`ids.txt` — CHISE IDS 파생 / 재정리 버전.

## Source And Licensing

- **프로젝트**: CJK VI Database
- **공식 repo**: `https://github.com/cjkvi/cjkvi-ids`
- **Zenodo DOI snapshot**: 10.5281/zenodo.1181440 (2018-02-20)
- **수집일**: 2026-04-23 (GitHub shallow clone)
- **라이선스**: **GPLv2** (`ids.txt`, Copyright CJKVI Database 2014-2017,
  based on CHISE IDS Database)
- **Unicode 기준**: cjkvi 는 CHISE 를 따라가지만 릴리스 주기가 CHISE 보다
  느리다. Ext J 등 최신 블록은 없음.

### License Caveat

GPLv2. CHISE 와 같은 상속 라이선스. 정제본은 원본에서 derived 이므로
GPL 규정을 따라야 하며, canonical_v3 DB 에 통합 시 라이선스 표기를
상속해야 한다.

## File Inventory

```
db_mining/RE_cjkvi_ids/
  data/                    # git shallow clone
    ids.txt                # 메인 UCS IDS
    ids-ext-cdef.txt       # Ext C/D/E/F 추가 분 (별도 라이선스 가능)
    ids-cdp.txt            # CDP private-use
    ids-analysis.txt       # 분석 파생본
    hanyo-ids.txt          # 일본 汎用 한자 IDS
    ucs-strokes.txt        # UCS → stroke count 별도 테이블
    ws2015-ids-cdp.txt, ws2015-ids.txt, waseikanji-ids.txt
    README.md
  process.py               # → db_src 변환

db_src/cjkvi_ids/
  ids.jsonl                # ids.txt + ids-ext-cdef.txt 병합 정제본
  ids_primary.jsonl
  stats.json
```

**우리 정제본은 `ids.txt` + `ids-ext-cdef.txt` 만 ingest**. 나머지 파일은
특수 용도 (waseikanji, CDP) 라 현재 파이프라인에 불필요.

## 원본 포맷

한 줄 포맷 (CHISE 와 동일):

```
# Copyright (c) 2014-2017 CJKVI Database
# Based on CHISE IDS Database
U+03B1	α	α
U+2113	ℓ	ℓ
U+9451	鑑	⿰金監
```

- CHISE 와 거의 동일. SMP codepoint 도 `U+XXXX` 로 표기 (CHISE 의 `U-`
  포맷 아님).
- 주석 `#` 로 시작. `;` 아님 (CHISE 와 차이).
- Multi-alternate 있을 수 있으나 드묾.

## 정제본 Schema

`babelstone_ids` / `chise_ids` 와 동일:

```json
{"codepoint": "U+9451", "char": "鑑", "ids": ["⿰金監"]}
```

## 통계 (2026-04-23 local)

- **총 엔트리**: 88,937
- **unique codepoints**: 88,937
- **multi-alternate**: 3,502 (3.9%)

블록별 분포:

| Block | Entries |
|---|---:|
| CJK Ext B | 42,711 |
| CJK Unified | 20,976 |
| CJK Ext F | 7,473 |
| CJK Ext A | 6,582 |
| CJK Ext E | 5,762 |
| CJK Ext C | 4,149 |
| CJK Compat Supp | 542 |
| CJK Compat | 472 |
| CJK Ext D | 222 |
| Other | 40 |
| CJK Radicals Supp | 8 |

(Ext G / H / I / J 없음 — 최근 확장 미반영)

## 핵심 가치 / 한계

### 가치
1. **단일 파일 `ids.txt` 로 간결** — 빠른 prototyping / lookup 에 편리.
2. **T1 10,932 class 에 99.9% (10,919) coverage** — 우리 학습 universe
   에서 단독 최고. BabelStone 97.9%, CHISE 99.7% 보다 높음.
3. 안정적 release (2018 snapshot 이후 maintenance 지속).
4. GPLv2 명시.

### 한계
1. **최신 확장 (Ext G/H/I/J) 미포함** — 88,937 로 BabelStone 97,649 /
   CHISE 102,892 보다 작음.
2. CHISE 기반이라 radical-position 변형형 없고 canonical form 만.
3. `&CDP-XXXX;` entity reference 가 일부 entry 에 등장 (CHISE 유래).

## 3-소스 비교 요약

| 지표 | BabelStone | CHISE | cjkvi-ids |
|---|---:|---:|---:|
| Unique codepoints | 97,649 | **102,892** | 88,937 |
| Ext G/H/I/J | 있음 (J 없음) | **모두 있음** | 없음 |
| Multi-alternate | 9,525 (9.8%) | 103 (0.1%) | 3,502 (3.9%) |
| Region flag 메타 | **있음** (G/H/T/J/K/P/V/X) | 없음 | 없음 |
| Radical-position 변형 | **있음** (釒, 钅 등) | 없음 (canonical form) | 없음 |
| 라이선스 | 명시 없음 | **GPLv2** | **GPLv2** |
| T1 10,932 coverage | 97.9% | 99.7% | **99.9%** |
| e-hanja 76,013 coverage | 98.2% | 99.4% | **99.5%** |

**권장 조합**: Primary = **CHISE** (최대 coverage + 라이선스 명확 + Ext J
cover). Secondary = **BabelStone** (region-specific variant lookup). Backup
= **cjkvi-ids** (빠른 prototype). 3 소스 union = 99.6% of e-hanja.

## 재생성 명령

```bash
cd db_mining/RE_cjkvi_ids
git clone --depth 1 https://github.com/cjkvi/cjkvi-ids.git data
python db_mining/RE_cjkvi_ids/process.py
```

## 관련 문서

- [doc/17_CANONICAL_V3_PLAN.md](../doc/17_CANONICAL_V3_PLAN.md)
- [BABELSTONE_IDS_MANUAL.md](./BABELSTONE_IDS_MANUAL.md)
- [CHISE_IDS_MANUAL.md](./CHISE_IDS_MANUAL.md)
