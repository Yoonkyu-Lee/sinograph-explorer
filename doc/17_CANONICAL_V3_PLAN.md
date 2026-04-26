# canonical_v3 Plan — 다음 병합 계획

작성일: 2026-04-21 (초안), 2026-04-23 (Phase 1 완료 반영 후 slim 화).

**이 문서는 계획 / 우선순위 / 설계 결정만** 담는다. 실제 빌드 수행 현황,
테이블 스키마 확정치, coverage 수치, 병합 로그 등은
[../sinograph_canonical_v3/BUILD_STATUS.md](../sinograph_canonical_v3/BUILD_STATUS.md) 로 옮겼다.

배경: `sinograph_canonical_v2` 는 T1 10,932 class 만 top-level 집계. v3 는
**76k+ universe** 로 확장하면서 **같은 종류의 정보를 여러 소스가 부분적으로
제공** 하는 상태를 정리해 통합 schema 를 설계한다. 현재 **Phase 1 완료**
— IDS 3 소스 병합 (`characters_ids` 103,046) + 구조 label 병합
(`characters_structure`). 나머지 Phase 가 이 문서의 대상.

---

## 1. 구조 정보 30 종 — taxonomy (reference)

학습 모델과 **relevant 한 정보**만 나열. 일반 사전 정보 (예문 / 용법 / 교육용
급수 등 글자 식별과 직접 무관) 는 제외.

### A. 식별·색인 축 (identifier / index)
1. **Codepoint** — Unicode scalar (`U+XXXX`). 모든 소스의 primary key.
2. **Character literal** — 해당 codepoint 의 실제 글자.
3. **Block** — Unicode block 범주 (CJK Unified / Ext A-H / Compat / …).

### B. 구조 (structure) 축
4. **Radical (부수) — 214-way 분류** — 강희자전 214 부수 중 어느 것에
   속하는가. 214-way classification label.
5. **Radical position variant** — 부수가 해당 글자에서 취하는 form
   (예: 金 → 왼쪽 form 은 `釒` 또는 `钅`).
6. **Radical residual stroke count** — 총획수에서 radical 획수를 뺀 나머지.
7. **Total stroke count** — 글자 전체 획수.
8. **Radical stroke count** — radical 자체의 획수.

### C. 합자 (composition) 축
9. **Flat component list** — 부품 codepoint list (IDS 기호 없이 평면 나열).
10. **IDS decomposition tree** — 재귀 prefix notation (⿰釒監, ⿱木⿰木木…).
11. **IDS top-level IDC** — 트리 루트의 IDC 기호 (12-way: ⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻).
12. **Component position** — 각 부품 위치 (left/right/top/bottom/surround).

### D. 조자 원리 (etymology) 축
13. **Etymology type** — 한자 유형 (형성 / 회의 / 상형 / 지사 / 가차
    또는 pictophonetic / ideographic / pictographic / radical-phonetic).
14. **Semantic component** (의미 부품) — 형성자에서 뜻을 담당하는 부품.
15. **Phonetic component** (음 부품) — 형성자에서 발음을 담당하는 부품.

### E. 획 (stroke) 축
16. **Per-stroke type** — 각 획의 종류 태그 (㇒㇐㇑ 등, CJK Strokes).
17. **Per-stroke geometry** — SVG path 좌표.
18. **Stroke order sequence** — 획순.
19. **Per-stroke component membership** — 획이 어느 sub-component 에 속하는가.

### F. 이체·변이 (variant) 축
20. **Variant edges** — codepoint 간 "같은 글자" 관계 (simplified /
    traditional / semantic / z-variant / spoofing / dongja / bonja / sokja
    / yakja / goja / waja / tongja / kanji 등).
21. **Variant family membership** — connected component of variant 그래프.
22. **Canonical representative** — family 내 대표 codepoint.
23. **Variant relation type** — edge 위의 관계 종류 레이블.

### G. 보조 색인 (auxiliary index) 축
24. **Cangjie code** — 5-character 창힐 입력 코드.
25. **Four-corner code** — 4-digit + 1 check digit.
26. **Phonetic class index** — Unihan `kPhonetic`.
27. **SKIP code** — Halpern 의 4-part shape classification (KANJIDIC2).
28. **Stroke number / frequency rank** — JLPT / 常用 / 교육용 급수.

### H. 발음·의미 축 (학습 signal 은 약하지만 사전 조회 / 평가용)
29. **Readings** — 발음. 다국어 (mandarin / cantonese / korean / ja_on /
    ja_kun / hangul / 훈음 / vietnamese).
30. **Meanings** — 뜻풀이 (영어 / 한국어 / 일본어 / 중국어).

위 30 종 중 **Phase 1 (구조 A-C)** 은 완료.
**Phase 2 (D, G)**, **Phase 3 (E)**, **학습-외 병합 (F, H)** 이 남았다.

---

## 2. 다음 병합 계획

원칙 재정립: **현 단계의 목표는 "인식 모델에 들어갈 학습 signal 만 우선"**.
모델 class 자체가 codepoint 이므로 codepoint 로부터 deterministic 하게
파생되는 정보 (family_id / canonical_representative / 뜻풀이 등) 는 training
label 로 들어가도 새 정보량 0 → **학습에 기여 없음**. 이런 건 "추론 후
자전 조회 / 평가 metric" 용으로 분리.

### 2.1 Phase 1.5 — Component vocabulary (Level A+ 용, 예정)

**목적**: train_engine 의 Level A+ component multi-label head 가 소비할
**K-차원 multi-hot label 테이블** 생성. Level A 학습 후 정체 시 바로
붙일 수 있도록 Phase 1 과 Phase 2 사이에 선행 빌드.

**왜 필요한가**: 현재 Phase 1 label (radical / stroke / IDC) 은
"글자 전체의 전역 속성" 이다. Component head 는 한 단계 더 내려가
**"이 글자는 무슨 부품들로 이루어졌는가"** 를 multi-hot 으로 supervise
한다. 鑑 → `{金, 監}`, 鍳 → `{金, 监}` 처럼 **같은 부품을 공유하는 class
간 representation 유사도** 를 강제. Midpoint 에서 드러난 鑑 / 鍳 /
鐱 계열 top-5 붕괴를 직접 공격하는 label 이다. 자세한 모델측 설명은
[18_FINAL_PRESENTATION.md](18_FINAL_PRESENTATION.md) Slide 5 Level A+.

**입력 소스** (전부 canonical_v3 에 이미 있음):
- `characters_ids.ehanja_components_json` — e-hanja 평면 부품 list, 76,013 cp
- `characters_ids.primary_ids` — IDS tree (평면화해서 부품 list 도출), 103,046 cp

**vocab 선정 규칙**:
- 두 소스를 union → 각 부품 codepoint 의 출현 횟수 count
- 빈도 상위 **K = 512** 만 vocab 에 포함. 나머지 부품은 ignore mask
- K 는 hyperparameter. 초기 K=512 권장 (sparse 하지만 학습 안정성 좋음).
  학습 진행 후 K=1024 로 확장 여부 재검토

**출력 테이블**: `characters_components` (신규)

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `codepoint` | TEXT PK | `U+XXXX` |
| `component_ids_json` | TEXT | 이 글자의 부품 codepoint list (vocab 필터 전 원본) |
| `component_vocab_idx_json` | TEXT | vocab 안에 있는 부품의 idx list (학습 시 multi-hot 으로 전개) |
| `has_any_vocab_component` | INTEGER | 하나라도 vocab 에 걸렸으면 1 (label mask 용) |

**별도 산출물**: `component_vocab.json` — K-entry vocabulary
(idx → 부품 codepoint + frequency).

**coverage 목표**: e-hanja 76 k 에서 ≥ 95 %, 103 k 전체에서 ≥ 85 %
(IDS atomic 글자는 자기 자신이 부품이므로 자동 cover).

### 2.2 Phase 2 — 학습 aux label (선택, coverage 낮음)

Phase 1 로 기본 accuracy 확보 후 추가 고려. coverage 낮아 label mask 필요.

| 필드 | 타입 | 주 소스 | 103k coverage | 비고 |
|---|---|---|---:|---|
| **etym_type** (5-way: 형성/회의/상형/지사/가차) | classification aux | e-hanja `etymology.type` + MMH `etymology.type` (1:1 매핑) | **15.6%** (16,083) | 형성문자 학습 signal 강함. label mask 필요 |
| **phonetic_class_index** (Unihan 숫자 번호) | embedding lookup / class | Unihan `kPhonetic` | **21.8%** (22,456) | 동음이자 grouping. CJK Unified 편중 |
| **cangjie** (5-char alphabetic) | sequence aux | Unihan `kCangjie` | **28.3%** (29,189) | shape index |
| **four_corner** (4+1 digit) | classification / regression | Unihan `kFourCornerCode` | **16.4%** (16,916) | 외곽 모서리 shape |
| **SKIP code** (4-part Halpern) | classification | KANJIDIC2 `q_code` | ~13% (일본 한자만) | 보류 |

→ 도입 시 `characters_aux_index` 테이블 추가. **현 단계에서는 보류**.
Phase 1 결과로 Level A 학습 한 후 accuracy 정체 시 고려.

### 2.3 Phase 3 — 학습 aux label (stroke-level, Level B/C 진입 시)

| 필드 | 타입 | 주 소스 | 103k coverage | 용도 |
|---|---|---|---:|---|
| **component_position_tree** | hierarchical label | KanjiVG `kvg:position` + e-hanja `radical.variant` | KanjiVG 6,699 (6.5%) | attention supervision, Level B |
| **per_stroke_type** (㇒㇐㇑ 등) | sequence label | KanjiVG `kvg:type` | ~6% (6,699) | stroke-level RNN / seq2seq |
| **per_stroke_geometry** (SVG path + median) | regression / rendering label | KanjiVG + MMH `graphics.txt` | ~15% (union) | stroke-level synthesis |
| **per_stroke_component_membership** | hierarchical label | MMH `matches` + KanjiVG 중첩 g | ~10% | stroke-to-radical 연결 |

→ coverage 낮아 현 multi-task 학습에 주 signal 로 부적합. Level B
(attention) 또는 Level C (zero-shot stroke decoder) 실험 시만 쓸 테이블
`characters_stroke_level`.

### 2.4 학습 외 병합 — 평가 / 자전 조회 (Phase 1 과 병행 가능)

코드포인트로부터 deterministic lookup 으로 얻을 수 있는 정보. model 에게
label 로 주면 정보량 0. 별도 테이블로 분리해 training loop 가 건드리지
않도록 한다.

| 필드 | 용도 | 103k coverage 목표 | 병합 소스 |
|---|---|---:|---|
| **family_id** + **family_members** | **evaluation metric** (family-aware accuracy), **confusion pair mining** | canonical_v2 T1 (100%) 을 103k 로 확장 | canonical_v2 `variant_components` + Unihan kTraditionalVariant/kSimplifiedVariant/kZVariant/kSemanticVariant + e-hanja `getSchoolCom` 10 관계 |
| **canonical_representative** | inference-time display 정규화 (모델 鑒 예측 → 鑑 으로 show) | 동상 | canonical_v2 확장 |
| **readings_*** (mandarin/cantonese/korean/ja_on/ja_kun/vietnamese/…) | 자전 display | 이미 canonical_v2 `character_readings` (T1 100%) | 103k 로 확장 |
| **meanings_*** (영/한/일/중) | 자전 display | 이미 canonical_v2 `character_meanings` | 103k 로 확장 |
| **hunum / jahae** (한국어 훈음) | 한국어 자전 display | e-hanja `getHunum` 76k 100% | e-hanja ingest |
| **education_level / hanja_grade / JLPT / frequency** | UI 필터 | low coverage | 참고만 |

→ 테이블 분리: `characters_family`, `characters_readings`,
`characters_meanings`, `characters_display_meta`. 학습 스크립트는 조회 안 함.

### 2.5 제외 확정 (OCR 학습에 무관)

- **Readings / Meanings** → DB lookup 용 (위 2.3 의 `characters_readings` /
  `characters_meanings` 에 둠)
- **Education level / hanja_grade / JLPT / frequency** → 사용자 UI 필터용.
  학습 label 아님
- **Pinyin tone number vs 기호 normalize** → 무관

---

## 3. Phase 별 우선순위

1. **Phase 1** — **완료** ✅ (radical_idx + total_strokes + residual_strokes
   + ids_top_idc, 103k 의 99.9%+ 커버). train_engine_v3 Level A multi-task
   head 의 label 로 바로 사용 가능. 상세: [BUILD_STATUS.md](../sinograph_canonical_v3/BUILD_STATUS.md)
   Section 8.
2. **Phase 1.5** (Level A+ 진입 선행) — component vocabulary + multi-hot
   테이블. Level A 학습 후 accuracy 정체 / 鑑 · 媤 계열 confusion 재현 시
   바로 붙일 수 있도록 **Phase 1 학습 돌리는 동안 병행 빌드**.
   이유: 데이터는 전부 이미 `characters_ids` 에 있고 (ehanja flat + IDS
   flat), vocab 선정 + 테이블 생성만 하면 됨 → 학습 쪽에서는 Level A 결과
   보고 "지금 붙일지" 만 결정하면 된다. 대기 시간 = 0.
3. **Phase 2** (학습 보조, 선택) — etym_type / phonetic_class / cangjie /
   four_corner. coverage 15-28% 라 label mask 필요. Level A+ 에서도
   accuracy 정체 시 추가 고려.
4. **Phase 3** (Level B/C, 향후) — per-stroke 계열. coverage 6-15%. zero-
   shot decomposition / attention visualization 실험 진입 시.
5. **학습-외 병합** (Phase 1 과 병행 가능) — family,
   canonical_representative, readings, meanings. 학습 엔진에 영향 없음.
   `characters_family` 가 가장 급함 (family-aware accuracy metric 에 필요).

---

## 4. 이후 section 에서 다룰 것 (placeholder)

- `characters_family` 테이블 병합 전략 (canonical_v2 + Unihan variants +
  e-hanja getSchoolCom 의 10 관계 통합 규칙)
- `characters_readings` / `characters_meanings` migration 방식
- train_engine_v3 Phase 0 metadata JSON (class_index_v3.json) 스키마
- Family-aware accuracy evaluation 경로
- Phase 2 도입 시점 결정 기준 (Level A accuracy metric)

본 문서는 여기까지.
