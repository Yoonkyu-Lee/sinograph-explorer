# Stage 1 Engine — 사용자 가이드

이 엔진은 **한 글자 codepoint → 한 장의 OCR 학습용 PNG** 를 만든다. 어떤 글자를,
어떤 서체·필체로 그리고, 어떻게 꾸미고, 어떻게 찍히고 망가뜨릴지를 YAML config
하나로 지정한다.

## 한눈에 보기

```
 base_source        style                 augment          finalize
 ─────────────      ─────────────────     ────────────      ──────────
 글자 모양 만듦  →  배경·색·장식 입힘  →  촬영·열화 시뮬  →  256×256 PNG
 (어디서 가져올지)  (디자인 의도)         (물리 조건)
```

- **base_source**: 폰트나 SVG 획 데이터로 glyph mask (L-mode) 를 만듦
- **style**: 마스크 위에 배경·채움·외곽선·그림자·네온 등 시각 디자인
- **augment**: 회전·블러·JPEG 압축·저조도 등 실제 사진처럼 열화
- **finalize**: 최종 256×256 PNG

세 블록은 각각 YAML 의 `base_source`, `style`, `augment` 섹션. **style/augment
는 이미지 레벨로만 작동**하므로 base_source 가 font 든 SVG 든 동일하게 적용됨.

## 블록 1: base_source — 글자 어디서 가져올까

### 사용 가능한 소스 종류 (`kind:`)

| kind | 데이터 | 커버리지 | 특징 |
|---|---|---:|---|
| `font` | Windows 폰트 (46 CJK face) | 폰트마다 다름 (~100k) | **인쇄체**. 결정적 — 같은 폰트·글자 = 동일 결과. face 랜덤 선택 |
| `svg_stroke` | MakeMeAHanzi medians | 9,574자 | **붓글씨 느낌** 필체. 중국어 상용 |
| `ehanja_median` | e-hanja skeletonize 결과 | 16,329자 | 필체. 한국 교육용 + SMP rare (Ext B 등) |
| `kanjivg_median` | KanjiVG 중심선 | 6,703자 | 필체. 日 shinjitai + 가나 (히라가나·카타카나) |
| `mmh_stroke` | MMH outline polygon | 9,574자 | canonical 모양 충실. 약간 딱딱함 |
| `ehanja_stroke` | e-hanja outline polygon | 16,329자 | canonical 모양 충실. 필체 느낌 X |
| `multi` | 위 중 여럿을 가중치로 랜덤 | 조합됨 | **학습 데이터용 권장** |

**커버리지 union 18,982자** (3 SVG 소스 합집합). 블록별 분포는
[`out/coverage_report.json`](out/coverage_report.json) 참고.

### 언제 뭘 쓰나

- **필체 느낌이 필요**: `svg_stroke` / `ehanja_median` / `kanjivg_median` 중 해당 글자 커버 소스
- **인쇄체만 필요**: `font`
- **두 종류 섞어서 학습**: `multi` (font + 획 median 가중치 섞기)
- **canonical 모양 보존이 최우선**: `*_stroke` (outline 기반, 변형 자연스럽지 않음)

### SVG 소스의 획 변주 (stroke_ops)

필체 variation 을 만드는 핵심 파라미터. 각 획마다 독립 적용:

| op | 의미 | 주요 파라미터 |
|---|---|---|
| `endpoint_jitter` | 획 시작·끝점 이동 (붓이 닿은/떨어진 자리 흔들림) | `std_ratio`, `std_min`, `std_max` |
| `control_jitter` | 획 내부 waypoint 전체 이동 (경로 굴곡) | 위와 동일 |
| `width_jitter` | 획당 두께 변화 (필압) | `std`, `mean` |
| `stroke_rotate` | 획별 개별 회전 | `angle_std` (도 단위) |
| `stroke_translate` | 획 전체 평행이동 | `std_ratio`, `std_min/max` |
| `drop_stroke` | 확률적으로 획 생략 | `prob_per_stroke` — **라벨 훼손 위험, 평가 전용** |

**`std_ratio`** 는 "획 길이의 X%" — 점(丶)은 덜 흔들리고 긴 획은 많이 흔들림.
3 소스 (box 1024 / 1152 / 109) 가 **같은 ratio 공유 가능**. 권장값:
- `endpoint_jitter std_ratio: 0.03`
- `control_jitter std_ratio: 0.02`
- `stroke_translate std_ratio: 0.015`

## 블록 2: style — 디자인 의도

### 배경 (`background.*`)

| 레이어 | 효과 |
|---|---|
| `background.solid` | 단색 (기본 흰색) |
| `background.noise` | 랜덤 노이즈 |
| `background.gradient` | 그라디언트 (두 색 사이) |
| `background.stripe` | 줄무늬 |
| `background.lines` | 노트 종이 느낌 등간격 라인 |
| `background.scene` | `samples/backgrounds/` 의 실제 이미지에서 랜덤 crop (가장 실전 같음) |

> 배경은 순서대로 합성됨 — 보통 `solid` 기본 → 다른 것들을 `prob:` 로 확률적 오버레이.

### 채움 (`fill.*`)

| 레이어 | 효과 |
|---|---|
| `fill.solid` | 단색 |
| `fill.gradient` | 선형 그라디언트 |
| `fill.radial` | 방사 그라디언트 |
| `fill.stripe` | **(비추)** 획 내부 줄무늬 — 획 연속성 파괴 |
| `fill.contrast` | **배경 밝기 감지 → 사전 팔레트에서 대비색 선택**. 기본 가독성 보장 |
| `fill.hsv_contrast` | **권장**. HSV 난수 (채도 보장) + signed luma 대비 체크. 무한 색 variation |

`fill.hsv_contrast` 핵심 파라미터:
- `saturation: [0.5, 1.0]` — 채도 하한 (회색 배제)
- `min_contrast: 60` — 글자-배경 luma 차 최소값
- `max_attempts: 8` — fallback (black/white) 전 재시도 횟수

### 외곽선, 그림자, 글로우

- `outline.simple` / `outline.double` — 글자 테두리
- `shadow.drop` / `shadow.long` / `shadow.soft` — 드롭섀도, 긴 그림자, 부드러운 그림자
- `glow.outer` / `glow.inner` / `glow.neon` — 외부 발광, 내부 하이라이트, 네온 조합

### 획 두께 (`stroke_weight.*`)

- `stroke_weight.dilate` / `stroke_weight.erode` — glyph mask 자체를 팽창/침식

## 블록 3: augment — 촬영·전송 열화

25+ op 를 6 그룹으로.

### 기하 (글자 포즈·각도)

- `rotate` — 회전 (`angle: [-15, 15]`)
- `perspective` — 투시 왜곡 (`strength: [0.04, 0.25]`)
- `scale_translate` — 확대·이동
- `shear` — 전단

### 광도

- `brightness`, `contrast`, `gamma`, `saturation`, `color_jitter`, `invert`

### 열화

- `gaussian_blur` — 일반 블러
- `motion_blur` — 모션 블러 (카메라 흔들림)
- `gaussian_noise`, `salt_pepper_noise` — 센서 노이즈
- `jpeg` — 압축 품질 저하 (`quality: [20, 98]`)
- `downscale_upscale` — 해상도 다운샘플 후 업샘플

### 스캔 시뮬

- `paper_texture`, `ink_bleed`, `binarize`, `shadow_gradient`, `vignette`

### 카메라 결함

- `defocus` — 초점 나감
- `chromatic_aberration` — RGB 분산 (렌즈 색수차)
- `lens_distort` — 렌즈 왜곡 (배럴/핀쿠션)
- `low_light` — 저조도

### 비강체 변형

- `elastic` — 손글씨 같은 자연 왜곡 (Simard 2003). alpha ≤ 15 권장 (라벨 안전)

## Source-aware layer/op 게이팅

style / augment 의 각 단계는 **현재 샘플이 어느 소스에서 왔는지** 알아보고 스스로
건너뛸 수 있음. 예를 들어 필체(`svg_stroke` / `ehanja_median` / `kanjivg_median`)
샘플은 이미 per-stroke jitter 로 자연 떨림이 있으므로 **전체-이미지 warp 계열
(`elastic`, `ink_bleed` 등)은 건너뛰는 게 좋음**.

두 키 사용:

```yaml
- op: elastic
  alpha: [4, 8]
  prob: 0.3
  skip_if_kinds: [svg_stroke, ehanja_median, kanjivg_median]  # 이 소스면 스킵

- op: some_op
  only_if_kinds: [font]                                       # 이 소스에만 적용
```

`source_kind` 는 엔진이 `Context.source_kind` 로 주입. 값은 위 kind 문자열 그대로.
multi-source 는 **picked child 의 kind** 를 기록해서 샘플마다 다르게 게이팅됨.

## 파라미터 문법

모든 값은 세 가지 형태:

| 형태 | 해석 |
|---|---|
| `5` 또는 `0.3` | scalar — 그대로 사용 |
| `[lo, hi]` (숫자 2) | uniform 랜덤 샘플링 |
| `[A, B, C, ...]` (3+ 리스트) | 이산 랜덤 선택 |

색은 항상 `[R, G, B]` 길이 3. 색 **풀** 로 주고 싶으면 `[[R,G,B], [R,G,B], ...]`.

모든 레이어·op 에 공통 `prob: 0.3` 옵션 — 30% 확률로만 적용.

## 실행

### 단일 글자 (`generate.py`)

```bash
python generate.py 鑑 --config configs/full_random_v2.yaml --count 10
```

- `--count` samples 만큼 출력. 소스 list 가 여러 개면 `sources × count` 생성
- `--sources "malgun,batang"` 으로 폰트 필터 override
- `--seed 42` 재현성
- `--metadata` 샘플별 sidecar JSON (어떤 소스가 picked 됐는지 등)

### 말뭉치 단위 (`generate_corpus.py`)

```bash
python generate_corpus.py --config configs/full_random_multi.yaml \
    --total 5000 --strategy stratified_by_block --metadata
```

- `--total N` 원하는 총 샘플 수
- `--pool union | intersection | mmh | ehanja | kanjivg` — 어느 글자 풀에서 뽑을지
- `--strategy uniform | stratified_by_block` — 균등 or Unicode 블록별 quota
- `--block-weights-json '{"Hiragana": 2.0, "Ext_B_SMP": 0.3}'` — 블록별 가중치 override
- 출력: `out/corpus_<config>/{idx}_{notation}_{source}.png` + `corpus_manifest.jsonl`

## 핵심 Config 파일

| Config | 용도 |
|---|---|
| `configs/full_random_v2.yaml` | font 단독, full style + augment |
| `configs/full_random_multi.yaml` | **프로덕션 권장**. multi-source + full style + augment |
| `configs/multi_source_handwriting.yaml` | multi-source 단순 config (style 간단) |
| `configs/ehanja_median_ratio.yaml` | e-hanja 단일 소스, ratio 기반 jitter (비교용) |
| `configs/stroke_demo/handwriting.yaml` | MMH 단일, 필기체 레퍼런스 |
| `configs/stroke_demo/stroke_stress.yaml` | `drop_stroke` 포함 — 평가 전용 |
| `configs/style/*.yaml`, `configs/augment/*.yaml` | 개별 블록 preset (ablation용) |

## Multi-source config 구조

```yaml
base_source:
  kind: multi
  fallback: font              # 글자를 어떤 소스도 커버 못 하면 fallback
  sources:
    - kind: font              # 각 항목이 자체 config
      filter: all
      weight: 5.0             # 가중치 (정규화됨, 5.0 = 50%)
    - kind: svg_stroke
      stroke_ops: [...]
      weight: 2.0
    - kind: ehanja_median
      width_scale: 1.5
      stroke_ops: [...]
      weight: 2.0
    - kind: kanjivg_median
      width_scale: 1.7
      stroke_ops: [...]
      weight: 1.0
```

**글자별 가용 소스 자동 필터링**: 해당 글자가 예를 들어 MMH 에만 없으면 MMH 그룹이
pool 에서 자동 제외되고 남은 가중치가 재정규화됨.

## 캔버스 사이즈 개념

- **CANVAS = 384**: 내부 작업 캔버스 (style/augment 가 여기서 돌아감)
- **PAD = 80**: glyph 를 캔버스 가장자리에서 떨어뜨리는 여백. 회전·perspective headroom
- **OUTPUT = 256**: finalize 시 center-crop 최종 크기

즉 glyph 는 약 224×224 로 그려지고 384×384 캔버스 안에서 augment 를 받음. 최종
256×256 으로 crop.

## 확장 — 새 기능 추가하려면

| 원하는 것 | 파일 | 패턴 |
|---|---|---|
| 새 배경 스타일 | `scripts/background.py` | `@register_layer("background.xxx")` 함수 추가 |
| 새 채움 방식 | `scripts/fill.py` | 동일 |
| 새 augment | `scripts/augment.py` | `apply_pipeline` 이 자동 등록 |
| 새 획 변주 op | `scripts/svg_stroke.py` | `@_register_stroke_op(...)` |
| 새 데이터 소스 (SVG DB 등) | 어댑터 `load_xxx_data` + `discover_xxx_sources` in `svg_stroke.py`, kind dispatch in `generate.py` |
| 커버리지 분석 | `scripts/coverage_report.py` 에 소스 추가 |

모든 레이어는 `Context(canvas, mask, rng, char) → Context` 시그니처. 실제로는
`ctx.canvas` 또는 `ctx.mask` 만 건드리고 반환. 새 레이어를 쓰려면 config 에
`layer: background.xxx` 한 줄 추가하면 됨 (등록만 해두면 자동 resolve).

## 산출물 디렉토리

```
synth_engine_v2/
  scripts/
    pipeline.py, generate.py, generate_corpus.py
    base_source.py                    # font
    svg_stroke.py                     # median 기반 (MMH / e-hanja / KanjiVG 공용)
    outline_stroke.py                 # outline 기반 (mmh_stroke / ehanja_stroke)
    background.py, fill.py, outline.py, shadow.py, glow.py, stroke_weight.py
    augment.py                        # 25+ op
    coverage_report.py                # 소스 커버리지 분석
  configs/                            # 위 표 참고
  samples/backgrounds/                # background.scene 용 실제 이미지 풀
  out/                                # 생성 결과
```

## 용어 요약

- **glyph mask**: 글자의 흑백 실루엣 (L-mode). base_source 의 산출물
- **canvas**: style/augment 가 공유하는 RGB 작업 이미지
- **kind**: base_source 의 종류 지정자 (`font`, `svg_stroke`, ... , `multi`)
- **layer**: style 블록의 원소 (`background.solid` 등)
- **op**: augment 블록의 원소 (`rotate`, `jpeg` 등)
- **stroke_ops**: SVG 기반 소스 안에서 각 획에 적용되는 변주 op
- **std_ratio**: 획 길이에 비례한 jitter 강도 (3 소스 공용 값 사용 가능)
