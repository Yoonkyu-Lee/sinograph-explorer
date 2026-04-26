# canonical_v3 Build Status

작성일: 2026-04-23 초기.
이 문서는 **실제 병합 수행 기록 + 현재 구축 상태**를 담는다. 계획 /
우선순위 / 결정 사항은 [doc/17_CANONICAL_V3_PLAN.md](../doc/17_CANONICAL_V3_PLAN.md)
참조.

---

## 1. 빌드 현황 요약

| Table / Field | Rows | 빌더 | 상태 |
|---|---:|---|---|
| `characters_ids` | 103,046 | `scripts/30_build_ids_table.py` → `32_reselect_primary_ehanja.py` | ✅ Phase 1 |
| `characters_ids.ehanja_*` | 103,046 | `scripts/31_merge_ehanja_components.py` | ✅ Phase 1 |
| `characters_structure` (radical / total_strokes / residual_strokes) | 103,046 | `scripts/35_build_structure_table.py` | ✅ Phase 1 |
| `characters_family` | — | 미구현 | 계획 |
| `characters_readings` | — | 미구현 | 계획 |
| `characters_meanings` | — | 미구현 | 계획 |
| `characters_aux_index` (etym / cangjie / …) | — | 미구현 | Phase 2 보류 |
| `characters_stroke_level` | — | 미구현 | Phase 3 |

**현재 학습 Level A 에 필요한 aux label 전체 완료** (radical + stroke +
residual + ids_top_idc 의 4 head label, 103k 의 99.9%+ coverage).

---

## 2. Universe 정리

```
BabelStone IDS      : 97,649
CHISE IDS           : 102,892
cjkvi-ids           : 88,937
e-hanja manifest    :  76,013
─────────────────────────────
3 IDS union = characters_ids : 103,046
(3 IDS ∪ e-hanja 시)          : 103,367
```

**항등식: `characters_ids` = BabelStone ∪ CHISE ∪ cjkvi = 103,046**.

차집합:
- **IDS union ∖ e-hanja = 27,354 cp** — 대부분 Ext G/H/I/J (CHISE 단독
  Ext J 4,298 cp 포함)
- **e-hanja ∖ IDS union = 321 cp** — **CJK Radicals Supplement**
  (U+2E80-U+2EFF, `⺁ ⺂ ⺃ ⺅…`). 부수 자체 form 이라 재귀 decomposition
  불가능. 학습 primary class 로 부적합 → **의도적 제외**.

→ characters_ids 103,046 은 학습 가능 한자 universe 를 이미 포괄.

---

## 3. Sources ingested

각 소스가 제공하는 정보 카테고리 정리. 규모 = 해당 소스의 codepoint 수.

### 3.1 Unihan (~97k CJK Unified + Ext)
Unicode 공식 backbone. `db_src/Unihan/Unihan_txt/*.txt`.

| 카테고리 | 제공 필드 | 비고 |
|---|---|---|
| Radical 214-way | `kRSUnicode` = "부수.잔여획수" (예: "167.14") | 거의 전체 cover |
| Residual strokes | `kRSUnicode` 뒷부분 파싱 | |
| Total strokes | `kTotalStrokes` | |
| Radical strokes | (파생: 강희자전 부수표) | 직접 필드 없음 |
| IDS decomposition | `kIDS` | **우리 로컬 파일엔 없음** (확인: 0/100 샘플) |
| Cangjie code | `kCangjie` | |
| Four-corner | `kFourCornerCode` | |
| Phonetic class | `kPhonetic` | |
| Variant edges | `kSimplifiedVariant` / `kTraditionalVariant` / `kSemanticVariant` / `kZVariant` / `kSpecializedSemanticVariant` / `kSpoofingVariant` | edge 타입별 분산 |
| Readings | `kMandarin` / `kCantonese` / `kJapaneseOn` / `kJapaneseKun` / `kKorean` / `kHangul` / `kVietnamese` | |
| Meanings | `kDefinition` | 영어 |

**없는 것**: IDS tree (로컬 기준), component position, per-stroke geometry,
etymology type 분류.

### 3.2 e-hanja online (76,013 / 75,669 detail)
온라인 한국 자전 수집. SVG + JSON API + HTML 3축.

| 카테고리 | 제공 필드 | 커버리지 (detail 75,669 중) |
|---|---|---|
| Radical 214-way | `radical.char` (부수 문자) | **100%** |
| Radical position variant | `radical.variant` (예: 金 → 钅) | 56.7% |
| Radical name | `radical.name` (예: "쇠금部") | 100% |
| Radical etymology description | `radical.etymology` | 100% (참고용) |
| Residual strokes | `radical_strokes` (전용 필드!) | 100% |
| Total strokes | `total_strokes` | 100% |
| Flat component list | `shape.components` (IDS 기호 없는 flat 형태) | **99.2%** (≥2 부품) / 99.7% any |
| IDS tree | ❌ | — |
| IDC top-level | ❌ (파생 어렵음, position variant 에서 추론 가능) | — |
| Etymology type | `etymology.type` (형성 / 회의 / 상형 / 지사 / 가차) | **17.4%** |
| Etymology description | `etymology.description` (자연어) | 53.2% |
| Readings | `getHunum` (한국 훈음, tree.jsonl 100%) / `pinyin` (54.8%) | |
| Meanings | `getJahae` (한국어 다중 뜻풀이, 100%) / `english` (27.7%) | |
| Variant edges | `getSchoolCom.{dongja / bonja / sokja / yakja / goja / waja / tongja / kanji / hDup / simple}` | union 45% (dongja 39.1% + 기타) |
| Per-stroke geometry | SVG outline + median (`strokes_animated.jsonl` 16,329 / `strokes_medianized.jsonl` 16,329) | 21.5% (animated only) |
| Per-stroke type | ❌ | — |
| Stroke order sequence | (animated SVG 의 path 순서로 추출 가능) | 21.5% |
| Auxiliary index | `classification.{education_level / hanja_grade / name_use}` | 2.6-12.1% |

**강점**: radical + 획수 + flat components 를 76k 규모로 100% 제공.
**약점**: IDS tree 포맷 없음, per-stroke type (㇒㇐㇑) 없음, etymology type
커버리지 17.4% 로 제한적.

### 3.3 e-hanja mobile (`ejajeon_plain.db`)
모바일 앱 SQLite. 한국어 뜻/훈음/관계형 테이블. T1 10,932 기반.
**learning signal 로는 e-hanja online 이 superset 이라 대체 가능** — 단
일부 필드 (교육용 한자 분류, 한자검정 급수) 는 독자적.

### 3.4 MakeMeAHanzi (MMH) — 9,574 entries
Chinese 상용 한자 decomposition + stroke graphics.

| 카테고리 | 제공 필드 | 커버리지 |
|---|---|---|
| IDS tree | `decomposition` (예: `⿰釒監`) | **100%** (9,574/9,574) |
| IDC top-level | `decomposition[0]` 파생 | 100% |
| Flat components | `decomposition` 파싱 파생 | 100% |
| Etymology type | `etymology.type` (pictophonetic / ideographic / …) | **94.4%** (9,033) |
| Semantic component | `etymology.semantic` | 94.4% |
| Phonetic component | `etymology.phonetic` | 94.4% |
| Etymology hint | `etymology.hint` (영어 한 줄) | 94.4% |
| Radical (대표) | `radical` | 100% |
| Per-stroke geometry | `graphics.txt` 의 SVG `strokes[]` + `medians[]` 좌표 | 100% |
| Per-stroke component membership | `graphics.txt` `matches[]` (각 stroke 가 어느 sub-component 에 속하는지 index) | 100% |
| Readings | `pinyin` | 100% |
| Meanings | `definition` (영어 한 줄) | 100% |

**canonical_v3 에서 MMH 역할 축소** (3 IDS ingest 후):
- `decomposition` / `radical` / `pinyin` / `definition` → 제거 (다른 소스 대체)
- **유지 필드**: `etymology.phonetic` / `etymology.semantic` (단독 소스),
  `graphics strokes/medians/matches` (stroke-level 용)

### 3.5 KanjiVG — 6,699 codepoints, 11,662 SVG variants
일본식 자형 기준 stroke-level SVG + 계층 annotation.

| 카테고리 | 제공 필드 | 커버리지 |
|---|---|---|
| IDS tree | ❌ (대신 **hierarchical `<g>` tree**) | — |
| Hierarchical structure tree | 중첩 `<g>` 에 `kvg:element` + `kvg:position` + `kvg:phon` + `kvg:radical` 태그 | 100% |
| Component position | `kvg:position` 값 (left / right / top / bottom / kamae / etc.) | 100% |
| Per-stroke type | `<path kvg:type="㇒">` (CJK Strokes U+31C0–U+31EF) | 100% |
| Per-stroke geometry | `<path d="...">` SVG path | 100% |
| Stroke order sequence | path 순서 + `<text>` stroke 번호 | 100% |
| Per-stroke component membership | 중첩 `<g>` 구조에서 파생 | 100% |
| Radical | `kvg:element` with `kvg:radical` 태그 | 100% |
| Phonetic component | `kvg:phon` 태그 | 100% (있는 경우 — 형성자 전용) |
| Variant indicators | `kvg:variant` / `kvg:tradForm` / `kvg:original` | 부분 |

**강점**: **유일하게 per-stroke type + component membership + position 을
통합 제공**. IDS 를 명시하진 않지만 중첩 g 구조에서 IDS 트리 복원 가능.
**약점**: 6,699 로 coverage 좁음 (76k 의 8.8%).

### 3.6 CNS11643 (대만)
`db_src/CNS11643/Properties/` 에 평문 파일 분산.

| 카테고리 | 제공 필드 | 비고 |
|---|---|---|
| Radical | `CNS_radical.txt` | CNS → 부수번호 |
| Total strokes | `CNS_stroke.txt` | |
| Cangjie code | `CNS_cangjie.txt` | |
| Flat components | `CNS_component.txt` (숫자 ID 리스트, CNS 내부 부품 코드) | 독자적 internal component index |
| Per-stroke type (코드) | `CNS_strokes_sequence.txt` (획순 숫자 시퀀스) | 포맷 확인 필요 |
| Readings | `CNS_pinyin_1/2.txt` / `CNS_phonetic.txt` | |
| Unicode mapping | `MapingTables/Unicode/` | CNS → Unicode |

### 3.7 KANJIDIC2 — 일본어 상용·인명용 약 13k
XML 엔트리형 일본어 자전.

| 카테고리 | 제공 필드 | 비고 |
|---|---|---|
| Radical | `<radical><rad_value rad_type="classical"/>` | |
| Total strokes | `<stroke_count>` | |
| Four-corner | `<q_code qc_type="four_corner">` | |
| SKIP code | `<q_code qc_type="skip">` | **Halpern 4-part shape**, 다른 소스에 드묾 |
| Variant | `<variant var_type="*">` | 일본 JIS 중복 |
| Readings | `<reading r_type="{pinyin, korean_r/h, vietnam, ja_on, ja_kun}">` | |
| Meanings | `<meaning>` (영어 + 다국어 `m_lang`) | |
| Frequency | `<freq>` / `<grade>` / `<jlpt>` | |

### 3.8 CEDICT / MOE_REVISED_DICT / MOE_VARIANTS / TONGYONG_GUIFAN
- CEDICT: Chinese-English 사전. Readings + meanings 만.
- MOE_REVISED_DICT: 대만 교육부 사전. Lexical dictionary, 단자 엔트리 포함.
- MOE_VARIANTS: 대만 교육부 **이체자 사전**. Variant edges 전문.
- TONGYONG_GUIFAN: 중국 통용규범한자표. Character list + frequency rank.

이들 넷은 **lexical / variant / frequency** 정보 위주라 structure-aware
학습의 direct signal 은 아니고 **variant edges 및 frequency 보조** 용도.

### 3.9 BabelStone IDS — 97,649 codepoints
Andrew West 의 IDS database (Unicode 16.0, 2025-06-27). IDS tree 전용 소스.

| 카테고리 | 제공 필드 | 커버리지 |
|---|---|---|
| IDS tree | `ids[0]` (primary), `ids[1..]` (region alternates) | **100%** (97,649/97,649) |
| IDC top-level | 파생 | 100% |
| **Region flag per alternate** | G/H/T/J/K/P/V/X (IRG region codes) | **100%** — 유일 |
| Radical-position variant | IDS 안에 `釒` / `𤣩` / `⻖` 같은 position form 사용 | **100%** — 유일 |

Manual: [../db_src/BABELSTONE_IDS_MANUAL.md](../db_src/BABELSTONE_IDS_MANUAL.md)

### 3.10 CHISE IDS — 102,892 codepoints (3 IDS 중 최대)
CHISE 프로젝트 (Kyoto U. 계열). GPLv2.

| 카테고리 | 제공 필드 | 커버리지 |
|---|---|---|
| IDS tree | `ids` (대부분 단일) | **100%** (102,892/102,892) |
| IDC top-level | 파생 | 100% |
| Canonical component form | 金 / 王 / 阝 등 표준형 | 100% |
| CDP private-use reference | `&CDP-XXXX;` | 일부 (~2k entries) |

Manual: [../db_src/CHISE_IDS_MANUAL.md](../db_src/CHISE_IDS_MANUAL.md)

### 3.11 cjkvi-ids — 88,937 codepoints
CJK VI Database. CHISE 파생, GPLv2.

| 카테고리 | 제공 필드 | 커버리지 |
|---|---|---|
| IDS tree | `ids` | **100%** (88,937/88,937) |
| IDC top-level | 파생 | 100% |
| CDP private-use reference | 소수 | 부분 |

Manual: [../db_src/CJKVI_IDS_MANUAL.md](../db_src/CJKVI_IDS_MANUAL.md)

### 3.12 3 IDS 소스 비교 요약

| 지표 | BabelStone | CHISE | cjkvi-ids |
|---|---:|---:|---:|
| Unique codepoints | 97,649 | **102,892** | 88,937 |
| Unicode 버전 | 16.0 | 17.0 (Ext J 포함) | ~14.0 |
| Multi-alternate | 9,525 (9.8%) | 103 (0.1%) | 3,502 (3.9%) |
| Region flags | **있음** (G/H/T/J/K/P/V/X) | 없음 | 없음 |
| Radical-position 변형 | **釒 / 𤣩 / ⻖ 사용** | canonical (金 / 王 / 阝) | canonical |
| 라이선스 | 명시 없음 | **GPLv2** | **GPLv2** |
| T1 10,932 coverage | 97.9% | 99.7% | **99.9%** |
| e-hanja 76,013 coverage | 98.2% | 99.4% | **99.5%** |
| Union 3 소스 | — | — | **99.6%** of e-hanja, 99.9% of T1 |

---

## 4. 정보 종류 × 소스 coverage 매트릭스

### 4.1 Main coverage

| # | 정보 | Unihan | e-hanja online | MMH | KanjiVG | CNS11643 | KANJIDIC2 | canonical_v2 top-level |
|---|---|---|---|---|---|---|---|---|
| 1 | Codepoint | ~97k | 76k | 9.6k | 6.7k | 독자 ID | ~13k | 모든 entry |
| 2 | Character literal | O | 100% | 100% | 100% | via map | 100% | O |
| 3 | Block | (derivable) | — | — | — | — | — | — |
| 4 | **Radical 214-way** | kRSUnicode 99%+ | **100%** | 100% | 100% | O | O | T1 100% |
| 5 | Radical position variant | — | **56.7%** | (derivable) | **100%** (kvg:element) | — | — | — |
| 6 | Residual strokes | kRSUnicode 99%+ (parse) | **100%** (전용) | (derivable) | — | — | — | — |
| 7 | **Total strokes** | 99%+ | **100%** | 100% | 100% | O | 100% | T1 100% |
| 8 | Radical strokes | — | 100% (total − residual) | — | — | — | — | — |
| 9 | **Flat component list** | — | **99.2%** | 100% | 100% | **O (독자 index)** | — | — |
| 12 | **Component position** | — | (infer from radical.variant) | — | **100%** (kvg:position) | — | — | — |
| 13 | **Etymology type** | — | **17.4%** (한국 분류) | **94.4%** (영어 분류) | — | — | — | — |
| 14 | Semantic component | — | ≈radical (100% 근사) | 94.4% | **100%** (kvg:element w/ kvg:radical) | — | — | — |
| 15 | Phonetic component | — | 53.2% (description parse) | 94.4% | **100%** (kvg:phon) | — | — | — |
| 16 | **Per-stroke type** | — | — | — | **100%** (kvg:type) | 부분 (숫자 시퀀스) | — | — |
| 17 | **Per-stroke geometry** | — | **21.5%** (animated SVG) | **100%** (strokes + medians) | **100%** (SVG path) | — | — | — |
| 18 | Stroke order sequence | — | 21.5% | 100% (path array 순서) | 100% | 부분 | — | — |
| 19 | Per-stroke component membership | — | — | **100%** (matches) | **100%** (g 계층) | — | — | — |
| 20 | Variant edges | kSimplified/Traditional/Z/Semantic 부분 | **getSchoolCom 10-way** 총 union ~45% | — | kvg:variant/tradForm 부분 | — | — | variant_edges table |
| 21 | Variant family membership | — (edges 에서 파생) | (edges 에서 파생) | — | — | — | — | **variant_components 100%** |
| 22 | Canonical representative | — | — | — | — | — | — | **characters.canonical_representative** |
| 23 | Variant relation type | O (edge type 별 필드) | O (10 종) | — | — | — | — | **variant_edges.relation** |
| 24 | Cangjie | 95%+ | — | — | — | O | — | — |
| 25 | Four-corner | 83%+ | — | — | — | — | **100%** (q_code) | — |
| 26 | Phonetic class index | 82%+ (kPhonetic) | — | — | — | — | — | — |
| 27 | SKIP code | — | — | — | — | — | **100%** (q_code) | — |
| 28 | Stroke number / frequency | — | hanja_grade 8.3% | — | — | — | freq / grade / jlpt | — |
| 29 | Readings | **많음** (다국어) | hunum 100% / pinyin 55% | pinyin 100% | — | pinyin / phonetic | 다국어 | character_readings |
| 30 | Meanings | kDefinition | jahae 100% / english 28% | definition 100% | — | — | meaning 다국어 | character_meanings |

### 4.2 IDS tree 전용 (3 소스 비교)

| # | 정보 | BabelStone | CHISE | cjkvi-ids | canonical_v3 union | **∩ e-hanja 76k** | **∩ T1 10,932** |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Codepoints 수 | 97,649 | **102,892** | 88,937 | 103,046 | — | — |
| 10 | **IDS decomposition tree** | 100% (97,649/97,649) | **100%** (102,892/102,892) | 100% (88,937/88,937) | **103,046** | **99.6% (75,692)** | **99.9% (10,919)** |
| 11 | IDS top-level IDC (12-way) | 100% 파생 | 100% 파생 | 100% 파생 | 100% | 99.6% | 99.9% |
| Ext | Region flag per alternate | **있음 (G/H/T/J/K/P/V/X)** | — | — | BabelStone 만 보존 | — | — |
| Ext | Radical-position form (釒 / 𤣩) | **100%** (IDS 안에 직접) | — (canonical 금, 王) | — | 이중-form 보존 | — | — |
| Ext | CDP private-use ref (`&CDP-…;`) | 드묾 | 자주 | 드묾 | 2,038 rows 에 존재 | — | — |
| Ext | Ext J 커버리지 (Unicode 17) | ❌ | ✅ 4,298 단독 | ❌ | CHISE 전용 | — | — |
| Ext | 라이선스 | 명시 없음 | **GPLv2** | **GPLv2** | 혼합 (CHISE 주축 안전) | — | — |

MMH 은 **IDS 제공자 역할에서 제외** (9,574 로 76k 의 12.6% 만 cover, 3 IDS
소스 합쳐 99.6% 가 대체). MMH 는 `etymology.semantic/phonetic/type` +
`graphics strokes/medians/matches` 전용으로 유지.

### 4.3 `characters_ids` agreement level 분포

| Agreement level | 개수 | 비율 | 학습 aux weight 권장 |
|---|---:|---:|---:|
| `unanimous` (3 소스 완전 일치) | 72,149 | 70.0% | 1.0 |
| `structure_only` (top-IDC 동일, component 표기만 다름) | 21,550 | 20.9% | 1.0 |
| `singleton` (1 소스만 보유) | 4,469 | 4.3% | 0.7–0.9 |
| `disagree_atomic` (구조 자체 충돌) | 4,878 | 4.7% | 0 or 0.3 (mask/low-weight) |
| **안전 label pool (unanimous + structure_only + singleton)** | **98,168** | **95.3%** | — |

top-IDC 분포:

| IDC | 개수 | % |
|---|---:|---:|
| ⿰ (좌우) | 67,077 | 65.1% |
| ⿱ (상하) | 23,353 | 22.7% |
| ⿺ (받침 받침) | 3,630 | 3.5% |
| ⿸ (좌상 둘러) | 3,370 | 3.3% |
| ⿳ (상중하) | 1,380 | 1.3% |
| ⿵ (위 둘러) | 1,032 | 1.0% |
| (leaf) atomic | 961 | 0.9% |
| ⿴⿹⿲⿻⿷⿶⿽⿼ | 2,243 | 2.2% |

---

## 5. 겹침·부분 겹침 관계

### 5.1 완전히 겹치는 정보 (여러 소스에 중복)
- **Radical 214-way** — Unihan / e-hanja / MMH / KanjiVG / CNS11643 /
  KANJIDIC2 전부 제공. 소스 간 **거의 항상 일치** 하지만 미묘한 예외
  있음 (CJK Compat / Ext B 에서 Unihan 과 e-hanja 가 다른 부수 지정). →
  **통합 시 source precedence** 필요.
- **Total strokes** — 위 모든 소스 제공. 소스 간 일치 검증 가능.
- **Cangjie** — Unihan + CNS11643 제공. 보통 동일.
- **Readings (mandarin / cantonese / korean / ja_on / ja_kun)** — Unihan +
  e-hanja + MMH + KANJIDIC2 중복. 표기 차이 존재.

### 5.2 부분 겹침
- **IDS decomposition**: **3 IDS 소스 (BabelStone / CHISE / cjkvi) 가
  102k+ cover**. MMH tree 는 9.6k 로 완전 대체. Section 4.2 참조.
- **Component position**: **KanjiVG (kvg:position) 100% within 6.7k** 이
  유일한 explicit 소스. e-hanja `radical.variant` (56.7%) 는 부수 위치만
  간접 시사.
- **Per-stroke type / geometry / component membership**: **KanjiVG + MMH
  교차 커버리지가 협소** (각 6.7k / 9.6k, 겹침 더 적음). e-hanja
  animated SVG 21.5% 가 middle layer.
- **Etymology type**: e-hanja **한국 분류 (17.4%)** vs MMH **영어 분류
  (94.4% within 9.6k)**. 1:1 매핑 가능 (형성문자 = pictophonetic 등).
- **Variant edges**: Unihan (kSimplified/Traditional/Semantic/Z 등) +
  e-hanja getSchoolCom 10종 + MOE_VARIANTS + canonical_v2 기존 edges. 모두
  codepoint pair + relation type 쌍이지만 vocabulary 다름.

### 5.3 단일 소스
- **IDS tree format**: 3 IDS 소스 중 CHISE 주축 (102k). MMH 9.6k 는 중복.
- **Per-stroke type (㇒㇐㇑ 등)**: KanjiVG 전용.
- **Per-stroke component membership (matches)**: MMH `matches` 또는
  KanjiVG 중첩 g.
- **SKIP code**: KANJIDIC2 전용.
- **Four-corner**: Unihan + KANJIDIC2.
- **Phonetic class index (kPhonetic)**: Unihan 전용.
- **Radical position variant (金 → 钅)**: e-hanja 전용 (+ BabelStone IDS
  내부).
- **Residual strokes 전용 필드**: e-hanja `radical_strokes`.
- **Korean 훈음 (getHunum)**: e-hanja 전용.
- **Hanja grade (한자검정)**: e-hanja 전용.
- **JLPT / Japanese grade**: KANJIDIC2 전용.

### 5.4 canonical_v2 현재 상태
- **Top-level 병합된 것**: characters.radical / total_strokes /
  canonical_representative / enriched_representative, variant_components
  (family_members_json 포함), variant_edges, character_readings,
  character_meanings, source_presence.
- **Source_payloads 만** 안에 갇혀 있는 것: IDS decomposition (MMH) /
  etymology type / etymology semantic-phonetic / flat components (e-hanja
  shape.components) / e-hanja classification / Cangjie / four-corner /
  phonetic class / KanjiVG 계층 구조 / per-stroke geometry / per-stroke
  type.

---

## 6. IDS 병합 전략 및 실행

### 6.1 실측된 3-소스 일치 분포

```
Union 3 IDS DBs:    103,046 codepoints
  ├─ only 1 source:        4,469  (4.3%)
  └─ ≥2 source overlap:   98,577  (95.7%)
        ├─ unanimous:         72,149  (73.2%)
        └─ disagree:          26,428  (26.8%)
              ├─ component-only diff:  21,550  (81.5% of disagree)
              └─ structural diff:       4,878  (18.5% of disagree)
```

구조적 합의 ≈ **95%+**. 순수 구조 갈등은 ~5% (4,878건) — manual audit
가능한 규모.

### 6.2 불일치 3 유형 실 샘플

**유형 A — 구조 미세 차이 (4,878건)**
- U+3AC6 `㫆`: BabelStone `⿰方尒` vs CHISE `⿸㫃小` vs cjkvi `⿰方尒`
- U+5344 `卄`: `⿻一⿰丨丨` vs `卄` vs `⿻十丨`

**유형 B — Component identity 차이 (대부분, 21,550건)**
- U+321E0 `𲇠`: `⿰釒許` (BabelStone) vs `⿰金許` (CHISE/cjkvi) — 釒 vs 金
- U+2DE57 `𭹗`: `⿰𤣩畄` vs `⿰王畄`

**유형 C — Private-use reference (소규모)**
- `&CDP-8958;` 같은 CHISE 내부 entity reference

### 6.3 스키마

```sql
CREATE TABLE characters_ids (
    codepoint              TEXT PRIMARY KEY,
    primary_ids            TEXT,
    primary_source         TEXT,       -- 'chise' / 'cjkvi' / 'babelstone'
    ids_top_idc            TEXT,       -- ⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻ 또는 'leaf'
    ids_sources_bitmask    INTEGER,
    ids_alternates_json    TEXT,
    agreement_level        TEXT,
    has_struct_conflict    INTEGER,
    has_cdp_ref            INTEGER,
    -- e-hanja cross-check (31 script 가 추가)
    ehanja_components_json TEXT,
    ehanja_agreement       TEXT,
    ehanja_aligned_sources TEXT
);
```

### 6.4 단계적 병합 규칙
1. **단일 소스 cps** → primary = 그 소스 IDS. `singleton`.
2. **Unanimous cps** → primary = 그 IDS. `unanimous`.
3. **Component-only diff** → 다수 일치 IDS, `structure_only`.
4. **Structural diff** → 다수 일치, 전부 다르면 우선순위. `has_struct_conflict=1`.
5. **CDP reference** → `has_cdp_ref=1`.

### 6.5 Primary 선정: e-hanja-alignment priority (최종 규칙)

초안 30 은 coverage 우선 (CHISE > cjkvi > BabelStone) 이었으나, e-hanja
와 최대한 일치시키기 위해 전환:

```
1. e-hanja 가 있는 cp 에서 aligned IDS 소스가 하나 이상:
   → 그 중 cjkvi > chise > babelstone 우선순위로 primary
   → majority 규칙 override (예: 𤨒 CHISE+BabelStone `⿰𤣩恩` majority 여도
      cjkvi `⿰王恩` 이 e-hanja 와 맞으면 cjkvi 채택)
2. e-hanja 가 없거나 aligned 소스 없으면:
   → 기존 coverage priority fallback (CHISE > cjkvi > BabelStone)
   → Ext J 4,298 cp (CHISE 단독) 가 이 branch → CHISE 유지
```

### 6.6 실제 영향 (103,046 rows)

```
rows_changed:        56,795  (55.1%)  — e-hanja-aligned 쪽으로 이동
  → cjkvi:           53,571
  → chise:            1,731
  → babelstone:       1,493

rows_unchanged:      46,251  (44.9%)
  → no_ehanja_align: 42,097
  → already_match:    4,154
```

Ext J 4,298 cp 는 e-hanja 에 **0 건** (확인됨) → CHISE 유지 정당.

### 6.7 demo char 전후

| cp | char | before | after | e-hanja |
|---|---|---|---|---|
| U+9451 | 鑑 | chise `⿰金監` | **cjkvi** `⿰金監` (attribution) | matches_multi |
| U+24A12 | 𤨒 | chise `⿰𤣩恩` | **cjkvi `⿰王恩`** ← IDS 교체 | matches_cjkvi |
| U+24AE5 | 𤫥 | chise `⿰𤣩罍` | **cjkvi `⿰王罍`** ← IDS 교체 | matches_cjkvi |
| U+5AA4 | 媤 | chise `⿰女思` | chise `⿰女思` (unanimous) | unanimous |
| U+7553 | 畓 | chise `⿱水田` | chise `⿱水田` (unanimous) | unanimous |
| U+9669 | 险 | cjkvi `⿰阝佥` | cjkvi `⿰阝佥` | matches_multi |
| U+32B35 | 𲬵 (Ext J) | chise singleton | chise singleton | absent |

### 6.8 Build chain (실제 실행 순서)

| Step | 스크립트 | 역할 |
|---|---|---|
| 30 | `scripts/30_build_ids_table.py` | 3 IDS 소스 병합 → characters_ids (coverage priority 초안) |
| 31 | `scripts/31_merge_ehanja_components.py` | e-hanja components cross-check 컬럼 병합 (NFKC normalize 포함) |
| 32 | `scripts/32_reselect_primary_ehanja.py` | primary_ids 재선정 (e-hanja alignment 우선) |
| 31 (rerun) | `scripts/31_merge_ehanja_components.py` | 재선정된 primary 기준 ehanja_agreement 재계산 |

Lookup: `python sinograph_canonical_v3/scripts/40_lookup.py --char <글자>`.

---

## 7. 불일치 처리 — Homoglyph vs Lexicographic

소스 간 불일치는 **두 범주**로 나뉘며 해결 방식이 다르다.

### 7.1 문제 A — Homoglyph (자동 해결, NFKC)
- 정의: **같은 모양, 다른 codepoint**. Unicode 가 호환 목적으로 중복 할당
- 예: e-hanja 金 = U+F90A (CJK Compat) vs IDS 金 = U+91D1 (CJK Unified)
- 범위: e-hanja `shape.components` 7.4% (5,552 rows) 에 CJK Compat 사용
- 해결: Python `unicodedata.normalize("NFKC", ch)` → U+F90A → U+91D1 자동
- 결과: `disagree_all` 17,795 → 14,160 (3,635 교정). **1 회 처리로 종결**.

### 7.2 문제 B — Lexicographic disagreement (정책 결정)
- 정의: **같은 codepoint**, 다른 사전 관례
- 예: 龜 U+9F9C — Unihan 17획 vs e-hanja 16획 (획수 count 관례 차이)
- 예: 卢 U+5362 — Unihan radical 25 (卜) vs e-hanja 44 (尸)
- **정답 없음** — 둘 다 legitimate. Unicode/NFKC 도 섞지 않음
- 처리: **Unihan 우선 정책** + e-hanja 값을 `alt_sources_json` 에 보존

### 7.3 비교

| 범주 | 성격 | 해결 가능? | 처리 |
|---|---|---|---|
| **A. Homoglyph** | 같은 모양, 다른 cp | ✅ 자동, 1 회 | NFKC normalize |
| **B. Lexicographic** | 같은 cp, 다른 관례 | ❌ 자동화 불가 | Unihan 우선 + alt 보존 |

---

## 8. Phase 1 완료 — `characters_structure` (2026-04-23)

Build script: `scripts/35_build_structure_table.py`. Universe 103,046.

### 8.1 스키마
```sql
CREATE TABLE characters_structure (
    codepoint         TEXT PRIMARY KEY,
    radical_idx       INTEGER,   -- 1-214 (강희자전)
    total_strokes     INTEGER,
    residual_strokes  INTEGER,   -- total - radical_self_strokes
    sources_json      TEXT,      -- {"radical_idx":"unihan",…}
    alt_sources_json  TEXT       -- Unihan ↔ e-hanja 불일치 alt 값
);
```

### 8.2 Coverage 실측
```
radical_idx       :  99.92%  (102,960 / 103,046)
total_strokes     :  99.96%  (103,005 / 103,046)
residual_strokes  :  99.90%  (102,944 / 103,046)
ids_top_idc       : 100.00%  (이미 characters_ids 에 존재)
```

### 8.3 Fallback (Unihan 누락 → e-hanja 보강)
- radical: 48 건
- total_strokes: 7 건
- residual: 32 건

### 8.4 Lexicographic disagreement 실측
- radical: 429 건 (예: 卢 U+5362 — Unihan=25 vs e-hanja=44)
- total_strokes: 10,216 건 (예: 龜 — Unihan=17 vs e-hanja=16)

→ 학습은 **Unihan 값 사용**. e-hanja 값은 `alt_sources_json` 에 JSON 보존.

### 8.5 종합 Phase 1 완료 상태

**학습 aux label 4 종 전체 확보 (103k 의 99.9%+)**:

| label | 소스 | 컬럼 | 용도 |
|---|---|---|---|
| radical_idx (214-way) | Unihan + e-hanja | `characters_structure.radical_idx` | structure head |
| total_strokes (regression) | Unihan + e-hanja | `characters_structure.total_strokes` | complexity head |
| residual_strokes (regression) | Unihan 파생 + e-hanja | `characters_structure.residual_strokes` | residual head |
| ids_top_idc (12-way) | 3 IDS union | `characters_ids.ids_top_idc` | layout head |

→ train_engine_v3 Phase 0 / Level A 에 **바로 연동 가능**.

---

## 9. 재현 명령

```bash
# IDS 병합 체인
python sinograph_canonical_v3/scripts/30_build_ids_table.py
python sinograph_canonical_v3/scripts/31_merge_ehanja_components.py
python sinograph_canonical_v3/scripts/32_reselect_primary_ehanja.py
python sinograph_canonical_v3/scripts/31_merge_ehanja_components.py  # rerun

# Structure table
python sinograph_canonical_v3/scripts/35_build_structure_table.py

# Lookup
python sinograph_canonical_v3/scripts/40_lookup.py --char 鑑
```

---

## 관련 문서

- [doc/17_CANONICAL_V3_PLAN.md](../doc/17_CANONICAL_V3_PLAN.md) — 계획 / 우선순위
- [db_src/BABELSTONE_IDS_MANUAL.md](../db_src/BABELSTONE_IDS_MANUAL.md)
- [db_src/CHISE_IDS_MANUAL.md](../db_src/CHISE_IDS_MANUAL.md)
- [db_src/CJKVI_IDS_MANUAL.md](../db_src/CJKVI_IDS_MANUAL.md)
