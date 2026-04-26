# CHISE IDS Manual

이 문서는 [chise_ids](./chise_ids) 디렉터리의 구조와 활용법을 정리한다.
원천은 CHISE (Character Information Service Environment) 프로젝트의
IDS database — 일본 교토대 Kyoto Univ 계열 CJK 형상 연구 그룹이 관리.

## Source And Licensing

- **프로젝트**: CHISE
- **공식 repo**: `https://gitlab.chise.org/CHISE/ids`
- **GitHub mirrors**: `github.com/chise/ids`, `github.com/qundao/mirror-chise-ids`
- **수집일**: 2026-04-23 (gitlab.chise.org 에서 shallow clone)
- **라이선스**: **GPLv2** (명시적 open-source)
- **Unicode 기준**: Unicode 17.0 범위 (Ext A–J 전부 cover)

### License Caveat

GPLv2 는 우리 프로젝트 내부 학습용으로 문제 없지만, 모델에 IDS 정보를
embedding 해 배포할 때 share-alike 조건을 확인해야 한다. 학습 데이터
생성 / label 로만 쓰고 모델 자체는 GPL 비-의존인 구조이면 안전.

## File Inventory

```
db_mining/RE_chise_ids/
  data/                           # git shallow clone (~20k files, ~300 MB)
    IDS-UCS-Basic.txt             # U+4E00–U+9FFF (CJK Unified)
    IDS-UCS-Ext-A.txt
    IDS-UCS-Ext-B-1.txt ... -6.txt
    IDS-UCS-Ext-C.txt
    IDS-UCS-Ext-D.txt
    IDS-UCS-Ext-E.txt
    IDS-UCS-Ext-F.txt
    IDS-UCS-Ext-G.txt
    IDS-UCS-Ext-H.txt
    IDS-UCS-Ext-I.txt
    IDS-UCS-Ext-J.txt
    IDS-UCS-Compat-*.txt          # CJK Compat
    IDS-JIS-*.txt                 # JIS supplementary
    IDS-Daikanwa-*.txt            # Daikanwa dict
    IDS-CNS-1/2/3.txt             # CNS 11643
    IDS-CDP.txt                   # CHISE 내부 Private-Use
    IDS-CBETA.txt                 # CBETA 불전
    ChangeLog
  process.py                      # → db_src 변환 스크립트

db_src/chise_ids/
  ids.jsonl                       # 정제본 (UCS files only)
  ids_primary.jsonl               # primary IDS 만
  stats.json
```

**우리 정제본은 `IDS-UCS-*.txt` 만 ingest** (Unicode codepoint 기준). JIS /
Daikanwa / CNS / CDP / CBETA 파일은 origin-specific ID 를 쓰므로 codepoint
기준 ingestion 에 직접 사용 안 함.

## 원본 포맷

한 줄 포맷 (CHISE 관례):

```
;; -*- coding: utf-8-mcs-er -*-
U+4E00<TAB>一<TAB>一
U+9451<TAB>鑑<TAB>⿰金監
U-00020000<TAB>𠀀<TAB>𠀀
U-00020001<TAB>𠀁<TAB>⿻一&CDP-88CD;
```

주의사항:
- **SMP codepoint (U+20000 이상) 은 `U-00020000` 형식** 사용 (하이픈, 8자
  리). BMP 는 `U+XXXX` (플러스, 4-6 자리). 정제 시 둘 다 수용.
- `&CDP-XXXX;` 는 **CHISE 내부 Private-Use 참조** — Unicode 에 할당되지
  않은 sub-component 를 CHISE 자체 ID 로 표기. IDS 문자열 안에 entity
  reference 형태로 등장한다. 정제본에서는 그대로 유지.
- `;` 로 시작하는 줄은 메타데이터 / 주석, 정제 시 skip.
- Multi-alternate 는 드물지만 있음 (additional tab-separated columns 에
  위치 변형 등 기재).

## 정제본 Schema

### `ids.jsonl`

```json
{"codepoint": "U+9451", "char": "鑑", "ids": ["⿰金監"]}
```

multi-alternate 있는 드문 경우 `ids` 배열에 여러 개.

### `ids_primary.jsonl`

```json
{"codepoint": "U+9451", "char": "鑑", "ids": "⿰金監"}
```

## 통계 (2026-04-23 local, UCS files only)

- **총 엔트리**: 102,892
- **unique codepoints**: 102,892
- **multi-alternate**: 103 (0.1%)

블록별 분포:

| Block | Entries |
|---|---:|
| CJK Ext B | 42,719 |
| CJK Unified | 20,992 |
| CJK Ext F | 7,473 |
| CJK Ext A | 6,592 |
| CJK Ext E | 5,774 |
| CJK Ext G | 4,939 |
| **CJK Ext J** (Unicode 17.0 신규) | **4,298** |
| CJK Ext H | 4,192 |
| CJK Ext C | 4,160 |
| CJK Ext I | 622 |
| CJK Compat Supp | 542 |
| CJK Compat | 367 |
| CJK Ext D | 222 |

## 핵심 가치 / 한계

### 가치
1. **3 개 IDS 소스 중 coverage 최대** (102,892). 특히 **Ext J (4,298) 는
   CHISE 단독** — BabelStone / cjkvi 에 없음.
2. 라이선스 GPLv2 명시 → 법적 안전.
3. Daikanwa / JIS / CNS / CDP 보조 파일 포함 — 연구 용도로 확장 가능.

### 한계
1. `&CDP-XXXX;` entity reference 가 IDS 안에 섞여 있음 — Unicode-only 파
   이프라인은 이를 unknown 부품으로 취급하거나 CHISE CDP 표를 함께
   ingest 해 치환해야 함.
2. BabelStone 의 region flag (G/H/T/J/K/P) 같은 **region-specific 분기
   정보는 없음** — 한 codepoint 에 한 canonical IDS.
3. Radical-position 변형형 (釒, 钅 등) 없이 canonical form (金) 만 사용.

## 재생성 명령

```bash
# 원본 clone (이미 되어 있으면 생략)
cd db_mining/RE_chise_ids
git clone --depth 1 https://gitlab.chise.org/CHISE/ids.git data

# 정제
python db_mining/RE_chise_ids/process.py
```

## 관련 문서

- [doc/17_CANONICAL_V3_PLAN.md](../doc/17_CANONICAL_V3_PLAN.md)
- [BABELSTONE_IDS_MANUAL.md](./BABELSTONE_IDS_MANUAL.md) — 자매 소스
- [CJKVI_IDS_MANUAL.md](./CJKVI_IDS_MANUAL.md) — CHISE 파생
