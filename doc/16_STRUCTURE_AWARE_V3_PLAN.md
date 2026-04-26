# train_engine_v3 — Structure-Aware Hanzi Recognizer

작성일: 2026-04-21
배경: midpoint 완료 후 Track 2 final demo novelty 점검 과정에서, 현재 "vanilla
ResNet-18 + codepoint softmax" 구조로는 모델 novelty 가 약하다는 자체 진단.
User 가설 "구조 인지 가능한 NN 을 처음부터 재설계" 에 대한 평가 + 실행 계획.

---

## 1. 한자 식별의 구조적 난관 (일반 객체 인식과의 차이)

### Visual (픽셀 레벨)
- **1-stroke 위치차** (未↔末, 土↔士, 口↔日) — receptive field / early
  downsample 문제
- **동일 layout + 유사 복잡도 혼동** (险→途) — 좌우 분리 감지는 해도 radical
  식별 실패
- **속자 공유 → 확신도 하락** (鑑 top-1 35.6%, 2-5위 유사 radical chars)
- **Font 의 변환 불변성** — 같은 codepoint 가 serif/sans/명조/고딕 에 따라
  크게 달라짐. 일반 사물은 각도/조명이 변수, 한자는 font style 이 변수
- **같은 폰트에서 시각적 동일 이체자** — 鑑↔鑒 가 특정 폰트에선 식별 불가
  (원리적 한계)
- **Stroke 가 sub-pixel** — 14+ stroke 글자가 128² 에 들어가면 획 하나가
  0.5-1px. 일반 사물엔 이런 문제 없음

### Structural (의미 레벨)
- **합자 (composition) 구조** — 모든 한자는 IDC (⿰⿱⿲ 등) 12+4 종 중
  하나로 2D 레이아웃 기술 가능. 일반 사물은 이런 구조 없음
- **부수 (radical) = 의미 범주 색인** — 氵가 붙으면 물 관련, 木 이 붙으면
  나무 관련. 부수는 시각 패턴이 아니라 **semantic cluster indicator** 라
  radical label 하나만 추가해도 같은 부수 내 class 끼리 feature sharing 이
  의미-수준에서 일어남. (학습 시 실제 입력은 radical ID 하나 — 214-way
  classification label. 의미 문자열을 학습에 투입하는 게 아니라, "왜 이
  label 이 효과 있는가" 를 설명하는 용어임.) 일반 사물엔 없는 compositional
  semantics.
- **Phonetic component** — 많은 한자가 음 부품 공유 (鑑/鑒/鍳 다 監 공유).
  비슷한 발음 = 비슷한 부품 = 비슷한 모양 경향
- **Stroke order 표준성** — 일반 사물엔 없는 canonical temporal structure
- **Variant family 다대일 매핑** — 國/国/囯/圀 전부 "같은 글자"
- **Rare-common 불균형** — 常用 2,000 vs rare 20,000+, 각 class 의
  typographic 익숙함 격차

### 학습 레벨
- **Class 수 vs 샘플 수 비대칭** — 10,932 class × 500 sample 은 ImageNet
  (1000×1300) 대비 per-class 정보 적음
- **Label noise (이체자 간)** — 어떤 pair 는 "서로 틀리다" 라벨인데 시각적
  으로 거의 동일 → 학습 신호 모순
- **Visual neighbor zero-sum** — 𤨒↔媤 mutual regression 실험에서 직접
  관찰한 현상

---

## 2. User 가설 평가: "구조 인지 NN 으로 재설계"

### 동의 (강한 근거)
한자는 ideograph = 구조물. Pixel-only CNN 은 compositional regularity 를
data 만 보고 재발견해야 함 → non-data-efficient. Radical/IDS/component
정보를 explicit supervision 으로 주면 뉴럴넷이 composition 공간을 **강제로
공유** 하게 됨 → 같은 radical 끼리 feature 공유, 다른 radical 끼리 분리.
결과: 險↔途 같은 radical-mismatch 혼동 근본 감소.

HCCR 연구에서 이미 증명됨 (DenseRAN, RAN, TAL-tree 등). Radical-level
decoder 를 쓰면 **zero-shot 한자** (학습에 안 본 글자) 도 부품 조합으로
인식 가능. 이건 generic CNN 으론 불가능한 능력.

### 반론 / 주의 (codex 도 지적)
1. **Structure label 만으로 "구조 이해" 자동 생성 X** — multi-task head 단순
   추가는 표면적. Feature sharing 이 실제로 일어나도록 loss weight
   스케줄링 / early-stage 강제 / hierarchical head 등 설계 필요
2. **Radical 이 가장 ROI 큼** — IDS/component 전부 한 번에 도입보다 radical
   → structure type → component 순차 도입. Codex: "1st radical, 2nd
   structure, 3rd decomposition, 4th stroke count"
3. **Confusable pair batching 이 구조 label 만큼 중요** — 險/途 를 매 batch
   에 같이 넣는 curriculum 이 radical head 추가만큼 효과 있음
4. **원리적 한계는 구조로도 못 풀음** — 같은 폰트에서 렌더링이 동일한
   이체자는 이미지만 보고 구분 불가. 이건 family-aware metric 으로 우회
5. **"Data 70% + aug 20% + model 10%"** (이전 codex 인용) — 구조 label 이
   모델 쪽 개선인 만큼 상한이 제한적. 하지만 우리는 이미 data / aug 는
   튼튼함

**결론:** user 가설 맞음. 단 "scratch 부터 다시" 는 80% 과장. Radical + IDS
+ component 는 **label 추가만** 으로 대부분 달성 가능 (데이터 재생성 X).
재생성이 꼭 필요한 건 **stroke-level spatial supervision** 하나.

---

## 3. Namu Wiki 3문서 핵심

### IDS (한자 모양 설명 문자) — U+2FF0–U+2FFB, 12개
⿰ 좌우 / ⿱ 상하 / ⿲ 좌중우 / ⿳ 상중하 / ⿴ 둘러싸기 / ⿵ 위-둘러 / ⿶
아래-둘러 / ⿷ 좌-둘러 / ⿸ 좌상-둘러 / ⿹ 우상-둘러 / ⿺ 좌하-둘러 (辶
받침) / ⿻ 중첩. Unicode 15.1 에서 4개 더 추가 (⿼⿽⿾⿿).

**Prefix notation 재귀 트리** — 예: 鑑 = `⿰釒監`, 森 = `⿱木⿰木木`, 意 =
`⿳立曰心`. **Zero-shot 인식 가능성** 의 열쇠.

### 속자 — 학습 관점에선 무시 가능
속자 / 정자 / 약자 / 와자 등 이체자 세부 분류는 **모델 학습에 무관**.
각 이체자가 서로 다른 codepoint 에 대응되면 모델 입장에선 그냥 다른 class 일
뿐, 분류 체계를 label 로 넣을 이유가 없다. 변종끼리 연결은 **DB lookup /
family-aware rerank 단계에서만** 의미 (Unihan kSimplifiedVariant /
kTraditionalVariant / kZVariant 및 canonical_v2 `variant_edges` 가 이
용도로 이미 존재). 따라서 namu 속자 문서는 참고 배경 정보로만 두고, 구조
label 후보에선 제외.

### 부수 — 214 (강희자전)
위치별 8종: 변 / 방 / 머리 / 발 / 엄 / 받침 / 몸 / 제부수. Unicode Kangxi
Radicals (U+2F00-U+2FDF, 214자), CJK Radicals Supplement (U+2E80-U+2EFF,
위치 변형형). Unihan kRSUnicode = "부수.잔여획수" (예: 海 = 85.6).

---

## 4. 우리 DB / db_src 에 이미 있는 구조 정보 (**발견 — goldmine**)

### 기준 universe 선택 (2026-04-21 감사)

이전 초안은 T1 10,932 (ehanja 모바일 기반) 을 분모로 썼는데, 이는 현재
학습 class set 기준일 뿐 향후 확장 상한을 보여주지 못한다. 가장 큰 내부
universe 는 **e-hanja online**:

| 소스 | codepoints | 용도 |
|---|---:|---|
| `db_src/e-hanja_online/svg/` | **76,013** | SVG 획 자원 (manifest) |
| `db_src/e-hanja_online/tree.jsonl` | **76,013** | 훈음/자해/변이 관계 |
| `db_src/e-hanja_online/detail.jsonl` | **75,669** | 구조 필드 (radical, 획수, shape, etymology) |
| Unihan (CJK Unified + Ext) | ~97k | 표준 backbone |
| MakeMeAHanzi (MMH) | 9,574 | IDS tree + etymology (Chinese 상용 위주) |
| KanjiVG | 6,699 | stroke centerline + 계층 태그 |
| 현 T1 class_index | 10,932 | 학습 subset |

아래 표는 **분모를 e-hanja online detail 75,669 로 통일**.

### 제안 metadata schema 의 필드별 상세 (75,669 universe 기준)

| 필드 | 설명 | 최대 coverage 소스 | 75,669 중 coverage | canonical_v2 top-level 병합? |
|---|---|---|---|---|
| `idx` | 내부 class index | class_index.json (학습 subset 만) | — | N/A |
| `char` | Unicode literal char | `chr()` 파생 | 100% | ✅ `characters.character` |
| `radical_idx` | 강희자전 214 부수 번호 | **e-hanja `radical.char`** + Unihan `kRSUnicode` 교차 | **100%** (75,669/75,669 e-hanja) | ✅ `characters.radical` (T1 범위만) |
| `residual_strokes` | radical 뺀 잔여 획수 | **e-hanja `radical_strokes`** (전용 필드) | **100%** (75,668/75,669) | ❌ 미병합 |
| `total_strokes` | 총 획수 | **e-hanja `total_strokes`** + Unihan `kTotalStrokes` | **100%** (75,669/75,669) | ✅ (T1 범위) |
| `ids_decomp` | IDS 트리 (`⿰釒監` 포맷) | **MMH `decomposition`** (유일) | **12.6%** (9,574/75,669) | ❌ MMH payload 안에만 |
| `ids_top_idc` | IDS 최상위 IDC (12종) | MMH `decomposition[0]` 파생 | 12.6% | ❌ 파생 필드 |
| `components` (flat list) | 부품 codepoint 리스트 (IDS 기호 없는 flat 형태) | **e-hanja `shape.components`** | **99.2%** (75,082, ≥2부품) · 99.7% any | ❌ e-hanja payload 안에만 |
| `etym_type` | 한자 유형 (형성 / 회의 / 상형 / 지사 / 가차 등) | **e-hanja `etymology.type`** (한국 전통 분류) + MMH `etymology.type` (영어 분류) | e-hanja **17.4%** (13,168) · MMH **7.9%** (6,012/76k) | ❌ 미병합 |
| `phonetic` (음 부품) | 형성자의 음 부품 | MMH `etymology.phonetic` + e-hanja `etymology.description` 파싱 | MMH ~7.9% · e-hanja 53.2% free text | ❌ 미병합 |
| `semantic` (의미 부품) | 형성자의 의미 부품 | MMH `etymology.semantic` + **e-hanja `radical.char`** (radical 을 의미 부품으로 근사) | **100%** (radical 근사) | ❌ 미병합 |
| `family_id` | family 당 동일 ID | canonical_v2 + e-hanja `getSchoolCom` 통합 후 우리가 부여 | 전체 universe 가능 | ⚠️ 우리 생성 |
| `family_members` | variant family codepoint 리스트 | canonical_v2 `variant_components` + **e-hanja `getSchoolCom` (dongja/bonja/sokja/yakja/goja/waja/tongja/kanji/hDup)** | e-hanja union ~45% (dongja 39.1% + bonja 8.5% + 기타) | ✅ T1 범위만 canonical top-level |
| `cangjie` | 창힐 입력 코드 (5 영문) | Unihan `kCangjie` (유일) | Unihan ∩ 76k 대략 70-80% 추정 (확인 필요) | ❌ source_payloads.unihan 만 |

### e-hanja online 에만 있는 추가 구조 필드

e-hanja online 을 full ingest 하면 새로 얻는 asset:

| 필드 | 소스 | 75,669 coverage | 쓸모 |
|---|---|---|---|
| `radical.name` (부수 이름) | e-hanja detail | **100%** | label 확인용 |
| `radical.variant` (position 변형형, 金→钅) | e-hanja detail | **56.7%** | **IDC 위치 추론 signal** (radical 이 left-form 이면 ⿰ 가능성) |
| `radical.etymology` (부수 설명) | e-hanja detail | 100% | 참고 |
| `etymology.description` (자원 설명, 자연어) | e-hanja detail | 53.2% | phonetic/semantic 부품 파싱 가능 |
| `classification.education_level` | e-hanja detail | 2.6% | 한국 교육용 급수 |
| `classification.hanja_grade` | e-hanja detail | 8.3% | 한자검정 급수 |
| `classification.name_use` (인명용) | e-hanja detail | 12.1% | — |
| `pinyin` | e-hanja detail | 54.8% | 중국어 발음 |
| `english` gloss | e-hanja detail | 27.7% | 영어 뜻 |
| `getHunum` (훈음) | e-hanja tree (76,013) | 100% | "거울 감" 식 |
| `getJahae` (자해) | e-hanja tree | 100% | 다중 뜻풀이 |
| `getSchoolCom.dongja` (동자 변이) | e-hanja tree | **39.1%** | variant linking |
| `getSchoolCom.bonja` (본자) | e-hanja tree | 8.5% | 정자 |
| `getSchoolCom.sokja` (속자) | e-hanja tree | 3.9% | — |
| `getSchoolCom.yakja` (약자) | e-hanja tree | 0.4% | — |
| `getSchoolCom.goja/waja/tongja/kanji/hDup/simple` | e-hanja tree | 0.4-2.4% each | 고자/와자/통자/일본한자/중복/간체 |

### 그 외 보강 가능 필드

| 필드 | 소스 | 75,669 coverage | 쓸모 |
|---|---|---|---|
| `four_corner` | Unihan `kFourCornerCode` | Unihan ∩ 76k 대략 70%+ | 중국 전통 색인 |
| `phonetic_class` | Unihan `kPhonetic` | Unihan ∩ 76k 대략 60%+ | 동음이자 signal |
| `per_stroke_type` | KanjiVG `path kvg:type` | **8.8%** (6,699/75,669) | stroke 단위 type |
| `kvg_hierarchical_tree` | KanjiVG 중첩 `g` | 8.8% | radical/position/phon 트리 |

### 결정적 관찰 (75,669 universe 기준)

1. **e-hanja online 이 현재 가장 broad-coverage structural source** —
   radical + total_strokes + radical_strokes + shape.components (flat)
   네 필드를 **100% / 100% / 100% / 99.2% 제공**. 규모 76k 로 Unihan 과
   비슷한 범위.
2. **IDS tree 포맷** 은 여전히 MMH 단독 → **75,669 의 12.6% 만 커버**.
   Structure-aware 학습에서 IDS tree 를 쓰려면:
   - a) 12.6% 만 label, 나머지는 null mask
   - b) **CHISE IDS / babelstone IDS 추가 ingest** (~95%+ CJK Unified)
   - c) e-hanja `shape.components` 를 flat list 로 쓰고 IDC 는 radical
     position variant 또는 KanjiVG position 태그에서 **추론**
3. **e-hanja `etymology.type` 는 한국 전통 분류 (형성 / 회의 / 상형 / 지사
   / 가차)** — MMH 의 영어 분류 (pictophonetic / ideographic /
   pictographic) 와 거의 1:1 매핑 가능. 두 소스 합치면 aux label coverage
   증대.
4. **variant family 소스가 e-hanja `getSchoolCom` 으로 확장** —
   canonical_v2 에 현재 반영된 variant_edges 가 Unihan 중심이라면, e-hanja
   dongja 39% + bonja 8.5% 등 Korean-side 변이 관계가 풍부하게 추가됨.
5. **radical 을 semantic 부품으로 쓰면 100%** — 형성자의 의미 부품을
   radical 로 근사하면 ingest 없이도 "음-의 분해" aux 가능.
6. **canonical_v2 는 T1 10,932 만 top-level 집계** 하고 나머지 ~65k 는
   source_payloads 내부에만 있음. 76k 전체를 학습 universe 로 삼으려면
   **canonical_v3 빌드** 에서 e-hanja detail 필드들을 top-level 컬럼으로
   승격 필요.

### Priority 권장 (76k universe + e-hanja full ingest 시 ROI)

1. **radical_idx + total_strokes + residual_strokes** — e-hanja 100%. top.
2. **components (flat list)** — e-hanja 99.2% (≥2 부품). IDS tree 는
   아니지만 "어떤 부품 2-3개로 구성되는가" 자체가 강력한 multi-label.
3. **family_id + family_members** — canonical_v2 + e-hanja getSchoolCom
   통합으로 coverage 최대화.
4. **etym_type (5-way: 형성 / 회의 / 상형 / 지사 / 가차)** — e-hanja 17.4%
   + MMH 7.9% → union 20-25% 추정. 나머지 null mask.
5. **ids_top_idc (12-way IDC)** — MMH 12.6%. Mask 학습.
6. (선택) **CHISE IDS ingest** — IDS tree coverage 를 76k 의 85%+ 로
   확장. Zero-shot radical decomposition (Level C) 에 필수.
7. (선택) **e-hanja online full ingest → canonical_v3** — 모든 위 필드를
   top-level 컬럼으로 승격.

### 예시 (e-hanja detail 직접 조회)

```
鑑 (U+9451) ← e-hanja detail.jsonl
  total_strokes: 22, radical_strokes: 8
  radical: {char: 金, variant: 钅, name: 쇠금部, etymology: "흙에 덮여..."}
  shape.components: [金(쇠금), 監(볼감)]                     ← flat 2-component
  etymology.type: 형성문자
  etymology.description: "뜻을 나타내는 쇠금(金) 部 와 음을 나타내는 監 (감)이 합하여..."
  classification: {education_level: 고등용, hanja_grade: 3급II, name_use: 인명용}
  pinyin: jiàn, english: "mirror, looking glass. reflect."

鑑 + MMH dictionary.txt:
  decomposition: ⿰釒監                                      ← IDS tree (e-hanja 에 없음)
  etymology.type: pictophonetic, phonetic: 監, semantic: 釒

鑑 + KanjiVG 09451.svg:
  g[金 position=left radical=general] + g[監 position=right phon=監]
    └─ g[臣 left] + g[皿 bottom] + g[𠂉]
```

한 글자 단위로 세 소스를 교차하면 e-hanja 에서 radical + flat components
+ 한국 etym_type, MMH 에서 IDS tree + 영어 etym_type, KanjiVG 에서
position 태그 + stroke type 을 얻음. 76k universe 기준 e-hanja 가 주축,
MMH / KanjiVG 가 좁지만 깊은 보강.

이 한 글자에 이미 **radical + structure + components + per-stroke type +
etymology** 다 갖춰져 있음. 전체 T1 기준으로도 radical / family 는 100%,
IDS / etymology 는 55-59% 로 다시 수집할 필요 없음 (IDS 커버리지만 CHISE
ingest 로 선택적 확장).

---

## 5. Label 용어

- **Primary label**: codepoint (우리 지금 쓰는 10,932-way class)
- **Auxiliary labels / auxiliary supervision / multi-task labels**: radical,
  stroke count, IDS, etymology 등. **Head 별로 따로 loss 붙여 joint
  training** 하는 signal
- 형식적으로: **multi-label multi-task annotation**
- 이 중 character 간 "속한 family" 를 나타내는 건 **soft/shared label**
  (1-hot 아님, 확률 분산)
- Stroke-level spatial supervision (어느 픽셀이 어느 stroke) 은 **dense
  label / segmentation mask** — 이건 또 다른 카테고리

즉 user 의 "이미지 ↔ 코드포인트, 획수, 부수, 합자" 구조는 정확히
**multi-task multi-label supervision**.

---

## 6. 데이터셋 재생성 필요한가?

### 재생성 **불필요** (label 추가만)
- Radical head (Unihan 에서 lookup)
- Stroke count head
- IDS / structure type head (12-way IDC classification)
- Etymology type head
- Phonetic component head
- Family-aware soft label

기존 shard 5.47M 개에 metadata 만 붙여 class_index.json 확장하면 됨.
**0 재생성**.

### 재생성 **필요** (이 3가지는 기존 synth pipeline 의 output 확장 필요)
1. **Stroke-level spatial mask** — "이 픽셀이 몇 번째 stroke / 어느
   radical 에 속하는가" — KanjiVG SVG render 중 stroke id 를 mask channel
   로 같이 저장. 현 synth 는 단일 RGB 만 냄
2. **Component bounding box** — "이 박스가 좌측 radical, 이 박스가 우측
   phonetic" — attention supervision 용. IDS 트리 + 렌더 시 각 컴포넌트
   bbox 기록 필요
3. **Part-based augmentation** — 좌측 radical 만 blur / 우측 phonetic 만
   rotate 같은 구조 인지 augment — 현 synth 는 글자 단위 aug 만

이 3가지는 **novelty 상한**을 결정. 없으면 radical-level aux label 까진
가능하지만, attention-guided / stroke-RNN / zero-shot decomposition 은
불가능.

### 결정 포인트
- **Level A (label 만 추가)**: 현 best.pth + radical/stroke/IDS/family aux
  heads 로 2-3일 내 강력한 novelty 확보 — ROI 최고
- **Level B (bbox + stroke mask 추가 재생성)**: synth_engine_v4 로
  업그레이드, stroke id + part bbox 출력. 재생성 시간 ~반나절.
  Attention-guided model + zero-shot decomposition 가능
- **Level C (from-scratch NN 재설계)**: RAN/DenseRAN 류 radical-sequence
  decoder. 학습 완전 처음부터. 고난도 but 최고 novelty

---

## 7. 추천 train_engine_v3 방향

### Phase 0: Label 확장 (반나절, 재생성 없음)
`class_index.json` 을 **rich metadata format** 으로 확장:
```json
"U+9451": {
  "idx": 8234,
  "char": "鑑",
  "radical_idx": 167,
  "residual_strokes": 14,
  "total_strokes": 22,
  "ids_decomp": "⿰釒監",
  "ids_top_idc": "⿰",
  "components": ["釒", "監"],
  "etym_type": "pictophonetic",
  "phonetic": "監",
  "semantic": "釒",
  "family_id": 1245,
  "family_members": ["U+9451", "U+9452", "U+9373", "U+30FAB", ...],
  "cangjie": "CSIT"
}
```
→ 이 구조 하나로 모든 auxiliary head 에 signal 제공 가능. **재생성 0**.

### Phase 1: Structure-aware heads (Level A) — 2-3일
- Primary char head (10,932) + **radical head (214)** + **IDC head (12)**
  + **stroke regression** + **etymology head (3)** + **family soft label**
- Loss scheduling: 초반 aux 비중 0.3 → 말미 0.1
- **Confusable pair curriculum** (codex 강조) — 같은 radical 내 오답 top-k
  쌍 매 batch 에 강제 포함
- Backbone: 현 ResNet-18 warm-start + **ResNet-D stem + BlurPool + dilated
  stage 4** (이전 논의)

### Phase 2: Level B synth 업그레이드 (옵션, 1-2일)
synth_engine_v3 → v4 확장:
- 생성 시 component bbox 추출 (KanjiVG 트리 or MMH decomposition +
  mask_adapter 에서 렌더 중 tracking)
- output: `(image, label, component_masks[N×H×W], bbox[N×4])`
- 추가 shard format: `shard-XXXXX.npz` 에 `masks` / `bboxes` 필드

### Phase 3: Attention-guided model (Level B 활용)
- Backbone + **part-aware attention head** (component bbox 를 soft
  supervision)
- Component-level embedding → concat → char head
- Radical 쪽 attention map 이 실제로 radical 위치에 활성화하도록 spatial
  supervision

### Phase 4: (Optional) Level C — Radical sequence decoder
- IDS 트리를 sequence-to-sequence 로 예측 (attention decoder)
- 학습에 본 적 없는 한자도 부품 조합으로 top-k 후보 생성 가능
- **Zero-shot recognition** 데모 → 최강 novelty

### Phase 5: Evaluation + ablation
- baseline (v2 best.pth) vs v3.1 (Level A) vs v3.2 (Level B) vs v3.3 (Level C)
- 축: top-1, family-aware, confusable pair, **zero-shot accuracy** (Level
  C 때)

---

## 8. 다음 단계 제안

1. **Phase 0 label 확장 먼저** — 반나절. 여기까지는 확정 안건
2. **Phase 1 Level A** 까지는 고정 계획 (2-3일)
3. Phase 2 Level B 재생성 여부 = user 결정 — "완전 scratch" 정신 계승 여부
4. Phase 4 Level C = 시간 여유 보고 결정

"데이터셋부터 처음부터" 에 대한 내 답: **완전 scratch 는 불필요, 하지만
synth v4 로 spatial metadata 추가 생성은 novelty 상한 결정 요소**. User 가
zero-shot / attention visualization 까지 demo 에서 보이고 싶으면 Level B
필수, 아니면 Level A 로 충분.
