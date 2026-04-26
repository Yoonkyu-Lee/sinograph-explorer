# Sinograph Canonical DB v2 Plan

> **목적 재정의**: v2 는 **Sinograph Explorer 의 "자전 (lookup + variant graph
> + metadata)" backbone**. 광학 인식용 stroke geometry 는 `db_src/` 에서 v3
> 엔진이 직접 접근하며, v2 canonical 의 관심사가 아니다. v1 이 들고 있던
> `media.stroke_svg_paths`, `media.stroke_medians` 는 v2 에서 canonical 로부터
> **분리**되고, v2 의 mass 는 **e-hanja mobile → e-hanja_online 교체** 와
> **커버리지 확대** 에 집중한다.
>
> 상위 문서: `doc/04_SINOGRAPH_PROJECT_PLAN.md` (프로젝트), `doc/05`, `doc/06`
> (v1 설계), `doc/08`, `doc/09` (e-hanja_online 발견·리버싱)
>
> 이전 설계: [`sinograph_canonical_v1/`](../sinograph_canonical_v1/) 는 legacy 로
> 유지. v2 는 별도 워크스페이스 `sinograph_canonical_v2/` 에서 새로 빌드.

---

## 0. 데이터 배치 원칙 (v2 에서 명확화)

v1 의 "**intersection + extensions**" 철학을 데이터 구조 레벨에서 더 엄격히 반영한다. 단순 이분법이 아니라 **3-가지 케이스**로 세분:

### (A) 교집합 — 여러 소스가 같은 의미축·같은 분류체계의 값을 공급
예: `total_strokes` (Unihan/KANJIDIC2/e-hanja/MMH 전부), `readings.mandarin` (Unihan/KANJIDIC2/MMH)
- `core.<field>` 에 **중복 없는 단일 값** (authority pick)
- `provenance.<field>` 에 이 값이 나온 실제 소스 기록
- **`core_alternatives.<field>` 에 다른 소스가 준 값**을 소스별로 보존
  - 다른 값이 전부 같으면 = **corroboration** (신뢰도 신호)
  - 다른 값이 다르면 = **conflict** (audit 가능)

### (B) 합집합 − 교집합 — 한 소스만 제공하는 고유 정보
예: `korean_hun` (e-hanja_online only), `decomposition` (MMH only), `korean_romanized` (KANJIDIC2 only)
- `source_exclusive.<source>.<field>` 로 **소스 이름이 경로에 드러나게** 저장
- 경로가 곧 provenance. 별도 provenance 메타 불필요.

### (C) 같은 의미축이지만 분류체계가 다른 관계 데이터 — variant 가 대표적
예: Unihan `kTraditionalVariant` vs e-hanja `yakja` — 둘 다 variant 관계지만 분류축 자체가 다름
- **합치지 않는다.** 소스별 relation bucket 그대로: `variants.traditional` (Unihan), `supplementary_variants.ehanja_yakja`
- 소스 분류가 다른 것을 강제로 flatten 하면 의미 손실
- 그 위에 **enriched graph** (component-level 합침) 를 별도 뷰로 제공해 "모든 variant" 질의는 가능

### (C′) (C) 의 하위 케이스 — 같은 관계 쌍을 여러 소스가 공급
예: `鑑 ↔ 鑒` 을 Unihan `kSemanticVariant` + e-hanja `dongja` 둘 다 지지
- **edge-level corroboration**: 하나의 edge row 에 `sources: ["unihan", "ehanja_online"]` 로 여러 소스 기록
- edge 를 소스마다 중복 저장하지 않는다 (merge by `source_cp, target_cp`)
- 소스 수가 많을수록 confidence 높음 (downstream query 에서 filter 가능)

### 결과
어떤 필드를 보든:
1. "공통 pick 인가 (여러 소스가 동의 / dissent)" — `core` + `core_alternatives`
2. "한 소스 고유인가" — `source_exclusive.<source>`
3. "관계인데 분류축이 다른가" — 소스별 bucket
4. "같은 관계 쌍을 몇 개 소스가 지지하는가" — edge `sources` 배열

v1 의 `structure` / `media` / `core.definitions.korean_*` 같이 "이름은 일반 카테고리지만 실제로는 한 소스 고유" 였던 영역이 (B) 로 재분류되어 정리되고, 동시에 **소스 간 coverage·정확도 격차가 edge·필드 레벨의 corroboration signal 로 드러난다**.

---

## 1. 왜 v2 인가 — 한 줄

v1 은 `e-hanja` 데이터를 **모바일 앱 DB 기준 10,932자** 만 갖고 있다. 이제
`e-hanja_online` 이라는 훨씬 방대한 소스 (**71,716자, Ext B 42,711 포함**) 를
확보했고, **Korean-facing metadata + variant 관계 그래프 전부 6.5× 규모**로
교체·확장할 수 있다. 동시에 v1 의 canonical 에 끼어 있던 stroke geometry 를
분리해 "이 DB = 자전" 이라는 역할을 단일화한다.

---

## 2. 핵심 변경 요약

| 축 | v1 | v2 | 비고 |
|---|---|---|---|
| e-hanja | mobile app SQLite, 10,932 rows | **e-hanja_online**, 71,716 rows | **교체**. 훨씬 방대. `doc/09` 참고. |
| variant graph (enriched) | v1.1 기준 7,385 char 에서 성장 | **기존 + SMP Ext B** 대규모 엣지 | 온라인 `getSchoolCom` 이 `bonja/waja/goja/simple/kanji/tongja/yakja/sokja/hDup` 까지 제공 |
| Korean 읽기·뜻 | `e-hanja` mobile 10,932 | **e-hanja_online `getHunum` + `getJahae`** 71,716 | 온라인 전수 수집 필요 |
| KANJIDIC2 | 13,108 rows, Japanese on/kun + resolvable refs | 그대로 | 변경 없음 |
| Unihan | 102,998 rows, codepoint backbone | 그대로 | 변경 없음 |
| MakeMeAHanzi | decomposition + etymology + stroke media (9,574) | **decomposition + etymology 만** (9,574) | stroke media 는 v2 canonical 에서 제외 (OCR 전용 → db_src) |
| KanjiVG | canonical 미포함 (원래) | **canonical 미포함** (OCR 전용) | 변경 없음 |
| stroke media 섹션 | `media.stroke_svg_paths`, `media.stroke_medians` 있음 | **완전 제거** | db_src/ 에서 v3 엔진이 직접 접근 |
| canonical identity | `character` + `codepoint` | 그대로 | 변경 없음 |

---

## 3. KanjiVG / MakeMeAHanzi 필요성 판단

사용자 질문: "**KanjiVG 나 MakeMeAHanzi 는 필요 없을 수도 있으니 판단 바람.**"

### 3.1 KanjiVG → **v2 에 포함하지 않음** (v1 과 동일)
- KanjiVG 가 제공하는 것: SVG stroke geometry (획별 centerline + stroke-type 메타)
- **자전 관점 가치**: 없음. 읽기·뜻·표제자 목록·변이 관계 전부 **제공 안 함**.
- 이미 v1 에서 통합 대상 아니었음. 자전 목적에 맞지 않는 판단이 v1 단계에서 이미 내려져 있다.
- **OCR 합성 관점** 가치는 있음 — v3 엔진이 `db_src/KanjiVG/strokes_kanjivg.jsonl` 을 직접 import 해 `kanjivg_median` source 로 활용 중. 이 경로는 유지.
- 결론: **v2 canonical 에 통합 X**. db_src 에 저장된 채 v3 엔진 전용.

### 3.2 MakeMeAHanzi → **부분 포함** (역할 축소)
- MMH 가 제공하는 것 4가지:
  1. `decomposition` (IDS-like) — 자전 관점 **가치 있음**
  2. `etymology_type` / `etymology_hint` — 자전 관점 **가치 있음**
  3. `stroke_svg_paths` — OCR 합성용, 자전 관점 **가치 없음**
  4. `stroke_medians` — OCR 합성용, 자전 관점 **가치 없음**
- e-hanja_online 이 decomposition 이나 etymology 를 제공하지 않으므로 (1),(2) 는 **대체 불가 고유 가치**. 9,574자 제한 커버리지지만 그 범위 내에선 훌륭.
- 결론:
  - canonical v2 의 `structure` 섹션 유지 (MMH 기반 `decomposition`, `etymology_*`)
  - canonical v2 의 `media` 섹션 **완전 삭제** (stroke geometry 는 v2 scope 아님)
  - MMH staging adapter 의 역할: `dictionary.txt` 의 decomposition + etymology 만 가져오고, `graphics.txt` 는 **v2 에선 읽지 않음**

### 3.3 e-hanja (mobile) → **v2 에서 제거**
- online 이 모든 면에서 superset (`doc/09` 실측 비교표).
- 유일한 예외: 모바일 앱 내부 ID 체계 (`hSchool.id`, `busu.id` 등) — 이건 교육용 앱 내 참조라 canonical 자전에 가치 없음. 버리는 게 깔끔.
- 결론: v2 에서 mobile 완전 제거. `db_src/e-hanja/` 는 historical archive 로 둠.

---

## 4. v2 Scope — "자전 only"

### 포함
- character 1자 당 metadata: 읽기 (mandarin/cantonese/korean/japanese/vietnamese), 뜻 (en/ko), radical, total_strokes
- variant graph: canonical (Unihan) + enriched (+ e-hanja_online + KANJIDIC2 resolved)
- structural meta: decomposition, etymology (MMH 범위 내에서만)
- source provenance (어느 소스가 이 char 를 봤는지, 어떤 값이 어디서 왔는지)

### 제외 (v2 에서 의도적으로 빼는 것)
- stroke geometry (SVG path, medians, polygon) — OCR 전용. db_src 가 담당.
- 어휘/성어/복사(複詞) lexical tables — v3+ 검토
- frequency/tier 분류 — v3+ 검토 (Tongyong Guifan 등 도입 시)

---

## 5. e-hanja_online 수집 현황 및 부족분

`doc/09` 의 Phase 2 수집 결과 (2026-04-19 기준):

| 축 | 상태 | 위치 | 파일 크기·행수 |
|---|---|---|---|
| Geometry 전체 (SVG composite) | ✅ 완료 | (raw SVG 디렉토리) | 71,716 파일 |
| Animated 획 분해 | ✅ 완료 | `db_src/e-hanja_online/strokes_animated.jsonl` | 16,329 rows |
| Animated median 추출 | ✅ 완료 | `strokes_medianized.jsonl` | 16,329 rows |
| Manifest (type 분류) | ✅ 완료 | `strokes_manifest.jsonl` | 76,013 rows |
| **`getSchoolCom` 변이 그래프** | ❌ **미수집** | 필요: `ehanja_online_variants.jsonl` | 71,716 rows 예상 |
| **`getHunum` 훈음** | ❌ **미수집** | 필요: `ehanja_online_hunum.jsonl` | 71,716 rows 예상 |
| **`getJahae` 뜻 리스트** | ❌ **미수집** | 필요: `ehanja_online_jahae.jsonl` | 71,716 rows 예상 |

**v2 선행 작업 (Phase 2.5)**: 위 3개 endpoint 를 전수 수집해야 자전 metadata
교체가 완성된다. `tool.img.e-hanja.kr` 의 JSON API 는 세션 불필요 + rate limit
약함 (09번 문서) — 3~4시간 수집 예상. 실제 수집은 `db_mining/` 쪽에서 별도 담당.

---

## 6. v2 Canonical Record Shape

v1 대비 핵심 변경:
1. **`media` 섹션 완전 제거** (OCR 재료 분리)
2. **`structure` 섹션 제거** → MMH 고유 정보이므로 `source_exclusive.makemeahanzi` 로 이동
3. **`references` 섹션 제거** → `source_exclusive.<source>` 로 흡수 (이미 소스별로 분리되어 있었으나 이름이 어중간했음)
4. **`core.definitions.korean_*` / `core.readings.korean_romanized` 이동** → 한 소스 고유이므로 `source_exclusive` 로
5. **`provenance` 블록 신설** — core 필드가 어느 소스로부터 실제로 채워졌는지 메타 기록
6. **`source_flags.ehanja` → `ehanja_online`** (mobile 과 혼동 방지)
7. **`supplementary_variants` 에 4개 관계 추가** (`waja`, `goja`, `sokja`, `hDup`)

```text
CanonicalCharacter (v2)
  character
  codepoint
  source_flags
    unihan
    ehanja_online     ← name change (was "ehanja")
    kanjidic2
    makemeahanzi

  core                               ← 교집합 (여러 소스가 같은 의미축을 공급)
    radical                          ← Unihan ∪ KANJIDIC2 ∪ e-hanja_online busu
    total_strokes                    ← Unihan ∪ KANJIDIC2 ∪ e-hanja_online ∪ MMH
    definitions
      english                        ← Unihan ∪ KANJIDIC2 ∪ MMH
    readings
      mandarin                       ← Unihan ∪ KANJIDIC2 pinyin ∪ MMH pinyin
      cantonese                      ← Unihan (only, but 언어축 통일성으로 core 유지)
      korean_hangul                  ← e-hanja_online hSnd ∪ KANJIDIC2 korean_h
      japanese_on                    ← KANJIDIC2 ja_on ∪ Unihan kJapaneseOn
      japanese_kun                   ← KANJIDIC2 ja_kun ∪ Unihan kJapaneseKun
      vietnamese                     ← Unihan kVietnamese ∪ KANJIDIC2 vietnam

  provenance                         ← core 각 필드가 이 record 에서 picked 된 실제 소스 (single)
    radical: <source_name>
    total_strokes: <source_name>
    definitions.english: <source_name>
    readings.mandarin: <source_name>
    readings.cantonese: "unihan"
    readings.korean_hangul: <source_name>
    readings.japanese_on: <source_name>
    readings.japanese_kun: <source_name>
    readings.vietnamese: <source_name>
    # 채워지지 않은 필드는 키 자체가 없음

  core_alternatives                  ← 같은 필드에 대해 picked 이외의 소스가 준 값. corroboration / conflict 신호.
    total_strokes                    ← 예: picked=unihan:22, alternatives={kanjidic2: 22, ehanja_online: 22, makemeahanzi: 23}
      <source_name>: <value>
    readings.mandarin
      <source_name>: [values]
    readings.korean_hangul
      <source_name>: [values]
    # 값이 picked 와 일치 → corroboration
    # 값이 다름 → conflict (downstream 에서 audit)
    # alternative 가 아예 없음 → 해당 필드는 단일 authority 만 보유

  variants                           ← Unihan canonical backbone (고유지만 프로젝트 중심이라 top-level 유지)
    traditional
    simplified
    semantic
    specialized_semantic
    z_variants
    spoofing
    representative_form
    family_members

  supplementary_variants             ← 소스 접두사로 이미 명시됨. e-hanja_online 확장
    ehanja_yakja
    ehanja_bonja
    ehanja_simple                    ← "simpleChina" 축약 (source 키 `simple`)
    ehanja_kanji
    ehanja_dongja
    ehanja_tongja
    ehanja_waja                      ← 온라인 신규
    ehanja_goja                      ← 온라인 신규
    ehanja_sokja                     ← 온라인 신규
    ehanja_hDup                      ← 온라인 신규
    kanjidic2_resolved

  variant_graph                      ← canonical vs enriched 병렬 뷰
    canonical_family_members         (= variants.family_members, Unihan backbone 만)
    canonical_representative_form
    enriched_family_members          (canonical + supplementary edges)
    enriched_representative_form

  source_exclusive                   ← 합집합 − 교집합. 한 소스만 제공하는 정보를 소스 경로에 명시해서 담는 곳
    unihan
      rs_unicode
      kangxi
      irg_kangxi
      hanyu
      unihan_core_2020
    ehanja_online
      korean_hun                     ← getHunum `hRead` 그대로
      korean_explanation             ← getJahae 뜻 리스트 (array of {meaning, root_snd})
      hschool_id                     ← (있으면)
      busu_id
      hshape / current / theory / law / length
      svg_type                       ← "animated" | "static"
      stroke_count_animated          ← animated 한해
    kanjidic2
      korean_romanized               ← KANJIDIC2 only (한글 romaja)
      dictionary_refs
      query_codes
      unresolved_variant_refs        ← resolve 실패한 것
      codepoint_refs
      nanori
    makemeahanzi
      decomposition                  ← IDS-like
      etymology_type                 ← pictophonetic / ideographic / ...
      etymology_hint                 ← explanation string
      phonetic_component
      semantic_component

  source_payloads                    ← near-raw. 변환 중 손실된 정보까지 전부 보존 (audit / provenance fallback)
    unihan_raw
    ehanja_online_raw
    kanjidic2_raw
    makemeahanzi_raw                 ← dictionary.txt 부분만
```

### 핵심 차이 — 왜 이렇게 나뉘는가

| 축 | 어떻게 분류되나 |
|---|---|
| `core.*` | **두 개 이상 소스**가 같은 의미축·같은 분류의 값을 공급. 하나의 값만 저장, provenance + core_alternatives 로 다른 소스 동의·반대 관찰. |
| `core_alternatives.*` | core 필드에 대해 picked 외의 다른 소스가 준 값. 같으면 corroboration, 다르면 conflict. |
| `variants.*` / `variant_graph.canonical_*` | Unihan backbone (분류체계 C). 프로젝트 중심이라 top-level 유지. |
| `supplementary_variants.ehanja_*` / `.kanjidic2_resolved` | 분류체계가 Unihan 과 다른 variant view. 소스 접두사로 명시. |
| `variant_edges` (SQLite) / `canonical_variants.jsonl` | 각 edge 에 `sources` 배열 — 같은 관계 쌍을 몇 개 소스가 지지하는지. |
| `source_exclusive.<source>.<field>` | **단 한 소스** 만 공급하는 필드. 경로가 곧 provenance. |
| `source_payloads.*_raw` | 변환 중 손실 방지용 near-raw 보존. |

### 삭제된 섹션
```
media                                ← v2 에서 완전 제거 (OCR 재료)
  stroke_svg_paths
  stroke_medians

structure                            ← v2 에서 제거 → source_exclusive.makemeahanzi 로 이동
  decomposition
  etymology_type
  etymology_hint
  phonetic_component
  semantic_component

references                           ← v2 에서 제거 → source_exclusive.<source> 로 흡수
  unihan
  kanjidic2
  ehanja
```

---

## 7. Source Authority Model (v2)

### 7.1 `core.*` 필드 (교집합) — authority pick + provenance 기록

여러 소스가 같은 의미축을 공급하는 경우만 이 표에 들어감. pick 결과는 단일 값으로 `core` 에 저장하되 **실제로 어느 소스가 이 record 의 값을 채웠는지는 `provenance` 블록에 키별로 기록**.

| Field | Primary | Fallback 순서 |
|---|---|---|
| `core.radical` | Unihan | KANJIDIC2 classical → e-hanja_online busu |
| `core.total_strokes` | Unihan `kTotalStrokes` | KANJIDIC2 `stroke_count` → e-hanja_online `stroke_count_animated` → MMH |
| `core.definitions.english` | Unihan `kDefinition` | KANJIDIC2 meanings → MMH definition |
| `core.readings.mandarin` | Unihan `kMandarin` | KANJIDIC2 pinyin → MMH pinyin |
| `core.readings.cantonese` | Unihan `kCantonese` | — (Unihan-only 지만 언어축 통일로 core 유지) |
| `core.readings.korean_hangul` | **e-hanja_online hSnd** | KANJIDIC2 korean_h |
| `core.readings.japanese_on` | KANJIDIC2 ja_on | Unihan kJapaneseOn |
| `core.readings.japanese_kun` | KANJIDIC2 ja_kun | Unihan kJapaneseKun |
| `core.readings.vietnamese` | Unihan kVietnamese | KANJIDIC2 vietnam |
| `codepoint` | Unihan | `ord(character)` |

예시:
```json
"core": {
  "readings": {"korean_hangul": ["감"], "mandarin": ["jiàn"], ...}
},
"provenance": {
  "readings.korean_hangul": "ehanja_online",
  "readings.mandarin": "unihan",
  "total_strokes": "unihan"
}
```

### 7.2 `source_exclusive.<source>.*` 필드 (합집합 − 교집합) — 경로가 곧 provenance

| Field | 소스 경로 | 내용 |
|---|---|---|
| `korean_hun` | `source_exclusive.ehanja_online` | getHunum 의 `hRead` — 훈·음 문자열 |
| `korean_explanation` | `source_exclusive.ehanja_online` | getJahae 뜻 풀이 리스트 |
| `svg_type` / `stroke_count_animated` / `hschool_id` / ... | `source_exclusive.ehanja_online` | Korean-educational metadata |
| `korean_romanized` | `source_exclusive.kanjidic2` | KANJIDIC2 only 한글 romaja |
| `dictionary_refs` / `query_codes` / `nanori` / `unresolved_variant_refs` | `source_exclusive.kanjidic2` | KANJIDIC2 고유 |
| `decomposition` | `source_exclusive.makemeahanzi` | IDS-like |
| `etymology_type` / `etymology_hint` | `source_exclusive.makemeahanzi` | MMH 고유 |
| `phonetic_component` / `semantic_component` | `source_exclusive.makemeahanzi` | MMH 고유 |
| `rs_unicode` / `kangxi` / `irg_kangxi` / `hanyu` / `unihan_core_2020` | `source_exclusive.unihan` | Unihan 전용 reference |

원칙: 값이 한 소스에서만 나올 수 있으면 `core` 에 넣지 않는다. 경로 이름 (`source_exclusive.ehanja_online.korean_hun`) 이 이미 출처를 말해준다.

### 7.3 Conflict / Corroboration policy (`core` 필드 전용)

- authority 순으로 첫 non-empty 값을 pick → `core.<field>`. picked 소스는 `provenance.<field>` 에.
- picked 이외의 소스가 같은 필드에 공급한 값이 있으면 **전부** `core_alternatives.<field>.<source_name>` 에 소스별로 저장.
  - 값이 picked 와 일치 → silently corroboration (downstream 이 `len(alternatives)` 로 support count 계산)
  - 값이 다름 → 명시적 conflict (UI·audit 이 양쪽 노출 가능)
- `source_payloads.*_raw` 는 여전히 유지 (raw 형태가 normalize 이전의 엣지 케이스 audit 에 필요할 때).
- missing → 키 자체 부재 (빈 array 로 채우지 않음).

---

## 8. Variant Graph 정책

### Canonical (Unihan-only) — v1 과 동일
- relation: `kTraditionalVariant`, `kSimplifiedVariant`, `kSemanticVariant`, `kSpecializedSemanticVariant`, `kZVariant`, `kSpoofingVariant`
- 출력: `variants.*`, `variant_graph.canonical_*`

### Enriched — 대폭 확장
- v1.1 은 e-hanja mobile 10,932 에서 supplementary edges 5,905개 공급.
- v2 는 **e-hanja_online 71,716 에서 공급** — BMP 부터 Ext B 까지.
  - `doc/09` 실측: `鑑` 한 글자만 해도 mobile 1개 → canonical 5개 → online 7개. 평균 3~4× 엣지 증가 예상.
  - SMP Ext B 의 Korean-context 관계는 **어디서도 구할 수 없는 유일 소스**. 이게 v2 의 가장 큰 가치.
- KANJIDIC2 resolved refs: 변경 없음.
- 출력: `variant_graph.enriched_*`

**경계 조건**:
- enriched family ⊇ canonical family (v1 과 동일 불변식)
- supplementary edge 는 Unihan bucket 에 섞지 않음 (source-explicit 유지, 섹션 0 의 케이스 C)
- representative-form heuristic: v1 과 동일 (traditional/reference 신호 → metadata richness → lexicographic minimum)

### 8.3 Edge-level corroboration (섹션 0 케이스 C′)

`canonical_variants.jsonl` 과 SQLite `variant_edges` 에서 각 edge row 는 다음 구조를 가진다:

```text
VariantEdge (v2)
  source_character
  source_codepoint
  target_character
  target_codepoint
  relation_scope      ← "canonical" | "supplementary"
  relation            ← Unihan 관계명 또는 e-hanja 관계명 (source-explicit)
  sources             ← 이 edge 를 지지하는 소스 배열, 예: ["unihan", "ehanja_online"]
  source_relations    ← 소스별 원 관계명 매핑, 예: {"unihan": "kSemanticVariant", "ehanja_online": "dongja"}
```

정책:
- 같은 `(source_cp, target_cp)` 에 대해 **edge row 를 중복 저장하지 않는다.** 여러 소스가 이를 지지하면 기존 row 의 `sources` 배열에 append + `source_relations` 에 각자의 관계명 등록.
- 단 `relation_scope` 가 다르면 (canonical vs supplementary) edge 는 별도 row (의미축 자체가 다르기 때문).
- `sources` 길이가 곧 **confidence tier**:
  - 2+ → strong corroboration (여러 소스가 독립적으로 확인)
  - 1 → single-source (Unihan 만 또는 e-hanja 만 아는 관계)
- downstream query: "신뢰도 높은 variant 만" / "e-hanja_online 만 아는 희귀 variant" 등 필터 가능.

이 정책이 네가 짚은 케이스 — **"e-hanja_online 이 특정 범위에서 훨씬 정확하고 풍부한데 Unihan 은 불완전"** 을 데이터 구조로 노출한다:
- e-hanja_online 범위 안 char: sources 배열 길이 평균 2+ → 이중 확인된 edges 가 많음
- e-hanja_online 범위 밖 char: sources 배열 = `["unihan"]` 뿐 → single-source confidence
- Unihan 이 놓친 관계: sources = `["ehanja_online"]` 뿐, but supplementary scope 로 존재 (enriched graph 에 반영)

---

## 9. Build Pipeline

### 9.1 Pre-build (v2 진입 전 선행)

**이 단계는 `db_mining/` 쪽에서 수행.** v2 canonical build 는 이 산출물이 준비된 후 시작.

1. e-hanja_online `getSchoolCom` 전수 수집 → `db_src/e-hanja_online/variants.jsonl`
2. e-hanja_online `getHunum` 전수 수집 → `db_src/e-hanja_online/hunum.jsonl`
3. e-hanja_online `getJahae` 전수 수집 → `db_src/e-hanja_online/jahae.jsonl`

선행 조건: `doc/09` 의 codepoint enumeration + URL pattern + polite crawl 정책 그대로 사용. 예상 4~6시간.

### 9.2 v2 Canonical Build

`sinograph_canonical_v1/` 구조를 참고하되 **새 워크스페이스** 에서 새로 작성:

```
sinograph_canonical_v2/
  CANONICAL_DB_V2_MANUAL.md
  README.md
  schema/
    canonical_schema_v2.md
  scripts/
    build_canonical_db_v2.py
    analyze_canonical_db_v2.py
    lookup_canonical_db_v2.py
    compare_v1_v2.py               ← v1 vs v2 sanity diff
  staging/
    unihan.normalized.jsonl
    ehanja_online.normalized.jsonl     ← 신규, v1 의 ehanja.normalized 대체
    kanjidic2.normalized.jsonl
    makemeahanzi.normalized.jsonl      ← media 필드 제거된 축소 버전
  out/
    canonical_characters.jsonl
    canonical_variants.jsonl
    variant_components.jsonl
    sinograph_canonical_v2.sqlite
    build_summary.json
  tests/
    sample_characters.py
    variant_graph_checks.py
```

### 9.3 Stage A — Source Adapters (재작성 or 재사용)

| Adapter | 기반 | 변경점 |
|---|---|---|
| `unihan_adapter.py` | v1 그대로 | 없음 |
| `kanjidic2_adapter.py` | v1 그대로 | 없음 |
| `makemeahanzi_adapter.py` | v1 | **`media` 필드 출력 제거**. `dictionary.txt` 의 decomposition + etymology 만. |
| `ehanja_online_adapter.py` | **신규** | 입력: `strokes_manifest.jsonl` + `variants.jsonl` + `hunum.jsonl` + `jahae.jsonl`. 출력: staging JSONL. |

### 9.4 Stage B — Identity Resolution
v1 과 동일. codepoint 우선, literal fallback. multi-character lexical row 는 canonical merge 진입 차단.

### 9.5 Stage C — Variant Graph
- Unihan edges → canonical relations
- e-hanja_online edges → supplementary (`waja/goja/sokja/hDup` 포함 10 relation types)
- KANJIDIC2 resolved refs → supplementary

### 9.6 Stage D — Canonical Projection
Authority 표 (섹션 7) 대로 fill.

### 9.7 Stage E — SQLite Projection

v1 대비 변경:
- `character_media` 테이블 **삭제** (media 섹션 제거)
- `source_presence.ehanja` → `source_presence.ehanja_online` (컬럼 이름 교체)
- **`core_provenance` 테이블 신설** — `(codepoint, field_path, source_name)` 3-컬럼. core 각 필드별 picked source 기록.
- **`core_alternatives` 테이블 신설** — `(codepoint, field_path, source_name, value_json)`. picked 이외 소스들이 해당 필드에 공급한 값.
- **`source_exclusive` 테이블 신설** — `(codepoint, source_name, field_path, value_json)`. source_exclusive 섹션의 projection.
- **`variant_edges` 테이블 보강** — 컬럼 추가: `sources_json` (배열 JSON), `source_relations_json` (dict JSON), `support_count` (integer = length of sources). `(source_codepoint, target_codepoint, relation_scope)` 에 unique index 걸고 same-pair row 는 merge.
- 기존 `source_payloads` / `variant_components` / `character_readings` / `character_meanings` 는 그대로.
- `characters.data_json` 은 full canonical record (새 shape) 를 담으므로 consumer 코드는 JSON 경로만 교체하면 됨.

쿼리 예시 (variant 신뢰도 활용):
```sql
-- 鑑 의 모든 variant 를 confidence tier 와 함께
SELECT target_codepoint, target_character, relation, support_count, sources_json
FROM variant_edges
WHERE source_codepoint = 'U+9451'
ORDER BY support_count DESC, relation;
```

### 9.8 Build Summary (v2)
v1 의 `build_summary.json` 형식 유지 + 새 지표:
- `ehanja_online_record_count`
- `smp_rare_coverage_count` (Ext B/C/D/E 중 variant edge 갖는 char 수)
- `variant_edge_count_v2_delta` (v1 대비 증분)

---

## 10. Test / Acceptance Plan

### 10.1 Adapter 단위
- e-hanja_online adapter: 71,716 row 나와야 함 (manifest 카운트 일치). Hunum/jahae 매칭률 95%+.
- MMH adapter: 9,574 row. `media` 필드 없어야 함.
- Unihan / KANJIDIC2: v1 과 row count 동일.

### 10.2 Merge 단위
- v1 과 canonical row count 차이가 극소 (±100). Unihan 이 backbone 이라 대세 유지.
- `source_flags.ehanja_online = true` count ≈ **71,716** (v1 의 10,932 에서 6.5×)
- 하위 집합 관계: v1 `source_flags.ehanja=true` 10,932 ⊂ v2 `source_flags.ehanja_online=true` 71,716 (ID 매핑으로 확인)

### 10.3 Character-level spot-check
- `鑑` (U+9451):
  - v1 enriched family: 5
  - v2 enriched family: **≥ 7** 목표 (doc/09 의 온라인 실측)
- `媤` (U+5AA4) — 한국 고유:
  - v1 에 row 있지만 Korean hun/explanation 빈약
  - v2 에 hRead + jahae 풍부
- `乶` (U+4E76) — 한국 고유 SMP-인접:
  - v1 커버리지 한계적
  - v2 e-hanja_online 으로 풍부
- `𨰲` (U+28C32, Ext B):
  - v1 에 Korean-context 정보 0
  - v2 에 dongja/simple 관계 존재 (doc/09 실측)
- `學` / `学` / `斈`: canonical family 보존 불변. enriched family 는 동일 or 확장.

### 10.4 Variant graph checks
- canonical ⊆ enriched (불변식)
- deterministic representative_form
- supplementary edges 가 Unihan bucket 을 오염시키지 않음

### 10.5 v1 ↔ v2 diff (scripts/compare_v1_v2.py)
- 공통 103k char 에 대해 row-by-row:
  - `core.readings.korean_hangul` 전체 중 **Δ%** (v2 는 e-hanja_online 기반이므로 상당수 새 값)
  - v1 `core.definitions.korean_*` → v2 `source_exclusive.ehanja_online.{korean_hun,korean_explanation}` 이동 확인
  - v1 `structure.*` → v2 `source_exclusive.makemeahanzi.*` 이동 확인 (MMH 9,574 만 영향)
  - `variant_graph.enriched_family_members` size 증가 분포
  - `media.*` 완전 부재 확인 (v2)
- 출력: markdown/CSV diff report. v1 user 들에게 이 리포트를 공유.

### 10.6 원칙 준수 검증 (v2 고유)
- **아무 `core.*` 필드도 단일 소스 고유가 아님** — 각 필드에 대해 최소 2개 소스가 커버 가능함을 record 샘플 1000 개에서 확인. 만약 특정 필드가 사실상 Unihan-only (예: cantonese) 여도 표에 명시한 예외 범주인지 재확인.
- **`source_exclusive.<source>.*` 의 모든 필드는 그 source 만 유일한 공급자** — 예: `source_exclusive.makemeahanzi.decomposition` 에 든 값이 다른 소스에서도 나올 수 있으면 재분류 필요.
- **`provenance` 의 소스값이 `source_flags` 와 일치** — provenance 가 "ehanja_online" 이라고 쓴 record 는 반드시 `source_flags.ehanja_online = true` 여야 함.

### 10.7 Corroboration / Conflict 검증
- `core_alternatives` 채움 비율 — 샘플 1000 char 에서 `total_strokes`, `readings.mandarin` 이 최소 2개 이상 소스 값을 갖는 비율 확인. 낮으면 adapter 파싱 누락 의심.
- **Corroboration case**: `total_strokes` picked=22, alternatives 전부 22 → `support_count` (암묵적) = n+1. sample char 중 corroboration 비율 리포트.
- **Conflict case**: picked 값과 alternatives 값이 다른 record 목록 추출 — human audit 용 CSV.
- **Edge corroboration**: `variant_edges.support_count` 분포 히스토그램. 鑑/鑒/學 같은 알려진 common variant 는 `support_count ≥ 2` 여야. 신규 Ext B SMP variant (e-hanja_online 만 앎) 는 `support_count = 1` 이고 `sources = ["ehanja_online"]` 여야.
- **Single-source variant 경고**: `support_count = 1` 이면서 supplementary_scope 가 아닌 canonical_scope edge 는 Unihan 단독 주장. 수도 확인.

---

## 11. Migration / Deprecation 정책

### 11.1 v1 → v2 소비자 영향
- Sinograph Explorer app / lookup demos: **v1 의 `media.*`, `character_media` 테이블을 사용 중이면 경로 변경 필요**. OCR 합성은 이미 `db_src` 직접 접근 (v3 엔진) 이라 영향 없음.
- sinograph canonical DB 를 `.sqlite` 로 직접 import 하는 외부 코드: table 이름 `characters` / `variant_edges` 그대로, 컬럼 `ehanja` → `ehanja_online`.
- 기타 consumer 는 `doc/06` 의 backward compatibility 원칙 (기존 `variants.*` 의미 불변) 그대로.

### 11.2 v1 폴더 처리
- 삭제하지 않음. `sinograph_canonical_v1/README.md` 상단에 "superseded by v2, kept for reference" 배너만 추가.
- v2 로 의존 이관 완료 후 v1 의 `.sqlite` / `.jsonl` 은 archive 이동 검토 (v2 안정화 후 결정).

### 11.3 migration 타이밍
v2 는 **Pre-build (섹션 9.1) 완료 후** 착수. e-hanja_online dictionary endpoints 수집이 없으면 v2 build 의미 없음 — v1 의 ehanja mobile 을 그대로 쓰는 것과 대동소이한 결과가 나옴.

---

## 12. 작업 타임라인 (권장)

| 단계 | 예상 작업량 | 내용 |
|---|---|---|
| **Phase 2.5 (선행)** — e-hanja_online dictionary crawl | 4~6시간 crawl + 1일 정리 | `getSchoolCom` + `getHunum` + `getJahae` 전수 수집. `db_mining/` 내부에서. |
| Phase A — v2 schema v1 + workspace 생성 | 반나절 | `sinograph_canonical_v2/` 뼈대, `schema/canonical_schema_v2.md` 작성 |
| Phase B — adapter 재작성 | 1~2일 | `ehanja_online_adapter.py` 신규, `makemeahanzi_adapter.py` 축소, 나머지 재사용 |
| Phase C — build pipeline | 1일 | `build_canonical_db_v2.py`, Stage B~E |
| Phase D — SQLite export | 반나절 | `character_media` 제거, 컬럼명 교체 |
| Phase E — 테스트 + v1 vs v2 diff | 1~2일 | 섹션 10 의 전 테스트 |
| Phase F — 문서화 | 반나절 | `CANONICAL_DB_V2_MANUAL.md`, README 업데이트, v1 에 deprecation 배너 |

전체: e-hanja_online crawl 포함 ≈ **1주 (단독 진행 시)**. crawl 을 별도 background 로 돌리면 실동 작업은 3~4일.

---

## 13. 열린 질문 / 설계 보류

1. **e-hanja_online 에서 `getHunum` / `getJahae` 가 돌려주는 응답의 NULL/404 비율**. 71,716 중 일부는 graph 만 있고 dictionary 미응답일 수 있음. 실 수집 이후 확정.
2. **`simpleChina` 필드의 split 규칙** — 일부 응답이 다중 char 인 경우 처리. `doc/06` 의 mobile 기준 규칙 재적용 가능한지 확인.
3. **v2 의 SQLite 용량** — v1 은 `characters.data_json` 에 full JSON 저장해 용량 큼. v2 는 `media` 제거라 축소, 하지만 e-hanja_online payload 증가. 실측 후 필요시 columnar split.
4. **v3 엔진의 `source_tag` 연동** — v3 의 `ehanja_median` source 는 `db_src/e-hanja_online/strokes_medianized.jsonl` 을 직접 읽는다. canonical v2 는 이 파일에 관여 안 함. 분리 명확히 유지하되, canonical 에는 "char X 는 e-hanja_online animated 획 가용" 여부만 `references.ehanja_online.svg_type` 에 남기는 것으로 접점 유지.

---

## 14. 변경 이력

- 2026-04-19 — 초안. v1.1 (doc/06) + e-hanja_online 리버싱 완료 (doc/09) + e-hanja_online geometry 수집 완료 상태에서 v2 설계. KanjiVG/MMH 역할 판정 포함.
- 2026-04-19 — "교집합 vs 합집합−교집합" 원칙을 데이터 구조 레벨로 반영. `core` + `provenance` + `source_exclusive.<source>.*` 3-layer 로 재구성. v1 의 `structure` / `references` / 일부 `core.*` 가 source_exclusive 로 이동. 이름이 일반 카테고리지만 한 소스 고유였던 필드 전부 정리.
- 2026-04-19 — "여러 소스가 같은 필드·같은 관계 쌍을 공급하지만 품질·coverage 가 엇갈리는" 케이스를 3-way 정제. `core_alternatives.<field>.<source>` 신설 (corroboration/conflict 노출) + `variant_edges.sources` / `support_count` 신설 (edge-level confidence). 단일 authority pick 만으로 커버 못 하던 e-hanja_online ↔ Unihan 정확도 격차를 데이터 레벨에 반영.

---

## 부록 A — 1-line mental model

> **v1**: Unihan backbone × `e-hanja mobile 10,932` + KANJIDIC2 + MMH (w/ stroke media) → 자전 + OCR 재료
>         `structure.*`, `references.*` 등 일반 카테고리 이름에 한 소스 고유 정보가 섞여 있음.
> **v2**: Unihan backbone × **`e-hanja_online 71,716`** + KANJIDIC2 + MMH (structure only) → **자전 only**
>         OCR 재료는 `db_src/` 에 분리 (v3 엔진 소관).
>         **교집합 정보 = `core.*` (+ `provenance`로 소스 메타)**, **합집합−교집합 = `source_exclusive.<source>.*` (경로가 곧 출처)**.
