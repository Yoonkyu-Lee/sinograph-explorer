# Stage 1 Engine — v1 Status

## Summary

이 문서는 `synth_engine_v1/` (이전 이름: `synth_experiment_01/`)의 **v1 상태**를 기록한다.

엔진의 존재 이유를 한 줄로: **문자 literal 하나를 받아서, 같은 문자를
표현하는 가능한 한 많고 다양한 glyph PNG를 만들어내는 장치**.

v1 엔진은 **단일 파이프라인 + 세 개의 교체 가능한 축**으로 구성된다.

```
base_source (pluggable)   effect_stack (pluggable)    augment (pluggable)
──────────────────────    ────────────────────────    ──────────────────
FontSource(...)           clean                        geometric
  ├─ malgun.ttf            solid_inverse                photometric
  ├─ batang.ttc            outline_thin / thick         degradation
  ├─ simsun.ttc            double_outline               scan_sim
  ├─ msgothic.ttc          bold                         camera_sim
  └─ ... (CJK 전체)        drop_shadow / long_shadow
                           gradient_warm / cool
SvgStrokeSource(...)       neon_cyan / pink
  (미구현)                  stripe_30 / 90
                           noisy_bg
ProceduralSource(...)      rotated_italic / ccw
  (미구현)                  warp_a / warp_b

          ↓                         ↓                       ↓
     glyph mask            stylized RGB (384→256)       varied RGB
      (L-mode)                                    (augment.py 모듈로 별도 존재,
                                                   아직 CLI 결합 안 됨)
```

**핵심 원칙**: base_source에 "font"가 들어오는 것도 effect_stack에 "clean"이
들어오는 것도 그 축의 **하나의 값일 뿐**이다. 별도 파이프라인이 아니다.
"system font로 clean 렌더" = `generate.py 鑑 --effects clean`.

## Input / Output

### Input
- character literal (예: `"鑑"`)
  - notation `"U+9451"`, 정수 `0x9451`도 1:1 동치
- CLI 인자:
  - `--sources`: 현재 `font:all` / `font:<substr>` / `font:a,b` 지원
  - `--effects`: effect 이름 콤마 리스트 또는 `all`
  - `--out`: 출력 경로 (기본 `out/<notation>/`)

### Output
- `out/<notation>/<source_tag>.<effect_tag>.png`
  - 예: `out/U+9451/batang-0.clean.png`, `out/U+9451/malgun-0.neon_cyan.png`
- 256×256 RGB
- 라벨 = 입력 literal 자체 (라벨 잡음 0)

## 글자 이미지를 결정하는 변수

같은 글자인데 이미지가 서로 달라 보인다면 **어떤 축의 값이 다르기 때문**이
고, 같아 보인다면 모든 축의 값이 같기 때문이다. 이 절에서는 v1 엔진이
가진 축을 **세 부류**로 분류한다. 분류 기준은 "값을 얼마나 자유롭게 움직일
수 있는가":

1. **진짜 변수** — 사용자가 넓은 공간에서 값을 고름
2. **프리셋 선택자** — 하드코딩된 번들 N개 중 하나를 고름 (내부 수치는 frozen)
3. **잠재 변수** — 원리상 변수가 될 자격은 있지만 v1에서는 프리셋에 묶여
   frozen 상태

---

### 진짜 변수 (v1에 2개)

#### 글자 (character)
이미지가 **어느 문자인지**를 정하는 축. 출력의 정체성이고 OCR 라벨.
鑑 과 學 은 애초에 모양 자체가 다르다 — 서체·스타일과 무관한 차원.

값의 공간: 유니코드 한자 코드포인트 약 100,000개.

#### 서체 (typeface)
같은 글자를 **어떤 필체로 그리는가**의 축. 폰트 하나 = 한 디자이너가 이
글자를 어떻게 쓸지 내린 결정의 집합.

서체가 다르면 달라지는 것:
- 획 굵기 (Light → Black)
- 서체 계열 (고딕 vs 명조)
- 지역 표준 (한국 / 일본 신자체 / 중국 간체 / 중국 번체)
- 자간·폭 (UI / 본문 / 등폭)

서체는 글자의 **골격**이다.

값의 공간 (v1): 시스템 CJK 폰트의 모든 face. 鑑 기준 46개가 식별된다.
각 face 자체는 이미 확정된 디자인이지만, face 간 비교해보면 의미 있게
큰 스타일 공간을 이룬다.

---

### 프리셋 선택자 (v1에 1개)

#### 스타일 (style preset)
19개의 하드코딩된 시각 번들 중 하나를 고르는 **이산 선택**.

각 번들은 내부에 색 RGB, 블러 세기, 외곽선 굵기, 섀도 오프셋, 글로우 반경,
회전 각도 같은 수치를 **전부 상수로 박아놓은 상태**. 같은 번들을 두 번
고르면 이미지가 완전히 동일하게 나온다 — 단 하나의 예외가 `noisy_bg`의
배경 노이즈 패턴(랜덤 시드에 의존).

즉 스타일은 "진짜 variable"이 아니라 **"미리 만들어둔 19개의 look 중
하나 pick"** 에 가깝다.

19개 번들 이름:
- 색/채움: `clean`, `solid_inverse`, `bold`
- 외곽선: `outline_thin`, `outline_thick`, `double_outline`
- 그림자: `drop_shadow`, `long_shadow`
- 그라디언트: `gradient_warm`, `gradient_cool`
- 네온: `neon_cyan`, `neon_pink`
- 줄무늬: `stripe_30`, `stripe_90`
- 노이즈 배경: `noisy_bg`
- 기하 왜곡: `rotated_italic`, `rotated_ccw`, `warp_a`, `warp_b`

두 이미지에 서로 다른 번들을 골라도, 번들 **내부에서는** 아무것도 변하지
않는다. 예를 들어 `neon_cyan` 번들의 청록색 `(0,220,255)`은 다음 `neon_cyan`
이미지에서도 똑같이 `(0,220,255)`.

---

### 잠재 변수 (v1에서는 frozen, v2에서 풀릴 수 있음)

스타일 번들 **내부**에는 원리상 진짜 연속 변수가 될 자격이 있는 수치들이
숨어 있다. v1에서는 번들마다 특정 값으로 고정되어 사용자가 건드릴 수 없지만,
이 수치들이 풀리면 variation 공간이 폭증한다. 주요 잠재 변수:

- **색** — 전경색 RGB, 배경색 RGB, 그라디언트 시작·끝색, 글로우 색, 섀도 색
- **획 처리** — 외곽선 굵기, bold dilation 반경
- **그림자** — 오프셋 (x, y), 블러 시그마, 투명도
- **글로우** — 바깥 반경, 안쪽 반경, 글로우 색
- **패턴** — 줄무늬 두께, 각도, 대비
- **기하** — 회전 각도, 전단 계수, 투영 왜곡 강도, 왜곡 방향
- **배경 질감** — 노이즈 강도, 입자 크기

v1 기준으로는 이 모든 값이 19개 번들 안에 **frozen**. 다시 말해 스타일
축이 "19개 이산값"처럼 보이는 이유는 잠재 변수들이 값을 고정당하고 있기
때문이다.

---

### 변수도 아니고 잠재 변수도 아닌 것

다음은 엔진에 존재하지만 **이미지 간 차이를 만드는 역할이 아니다**.

- **해상도 / 캔버스 / 여백** — 모든 이미지에 동일 적용되는 시스템 상수.
- **글자 크기 자동 맞춤** — 서체 간 크기 편차를 없애는 정규화 알고리즘.
- **난수 시드** — 이론상 랜덤 축이지만 v1 스타일 대부분이 결정적. 유일한
  반응 지점이 `noisy_bg` 하나.
- **augment (미결합)** — 열화 시뮬 라이브러리. 네 번째 축이 될 잠재력은
  있지만 v1 엔진에 연결되지 않아 현재 이미지에 영향 없음.

---

### 요약

| 분류 | 축 | 값의 공간 | v1에서 자유도 |
|-----|----|----------|-------------|
| 진짜 변수 | 글자 | ~100k 코드포인트 | **열림** |
| 진짜 변수 | 서체 | 46 face (鑑 기준) | **열림** |
| 프리셋 선택자 | 스타일 | 19개 중 하나 | **이산 선택만** |
| 잠재 변수 | 색·블러·각도 등 스타일 내부 수치 | 각자 연속 공간 | **frozen** |
| 비(非)변수 | 캔버스·정규화·시드·augment | — | — |

v1 이미지 한 장의 실질적 고유 ID는 **(글자, 서체, 스타일번들)** 세 값의
튜플. 하지만 이 중 "진짜 유의미한 variation"을 만드는 건 사실상 앞 두 개고,
스타일은 19개 카탈로그에서 고르는 셈이다.

v2의 방향은 명확하다: **스타일 내부의 잠재 변수들을 풀어서** 이산 카탈로그
대신 **연속 파라미터 공간**으로 바꾸면 variation 공간이 곱으로 폭증한다.

## v1 구조

### 1. `base_sources.py` — 글리프 모양의 출처

`BaseSource` ABC를 선언하고 v1에 **FontSource** 한 종류를 구현.

```python
class BaseSource(ABC):
    def render_mask(self, char: str) -> Image.Image | None: ...
    def tag(self) -> str: ...

@dataclass
class FontSource(BaseSource):
    font_path: Path
    face_index: int = 0
    ...
```

- `render_mask`는 L-mode 384×384 마스크를 반환 (white=glyph)
- target box에 맞춰 **binary-search로 font size fit**
- `discover_font_sources(fonts_dir, char_filter)`로 `C:/Windows/Fonts`를
  훑어 literal을 커버하는 모든 face를 `FontSource` 리스트로 반환

**鑑 기준: 406 face 중 46 face가 cmap 보유 → FontSource 46개 생성.**

### 2. `effects.py` — 모양 → 스타일

`REGISTRY` 딕셔너리 하나에 effect 함수를 `@register(name)`으로 등록.
각 함수 시그니처는 `(mask, rng) -> RGB 256×256`.

```python
REGISTRY: dict[str, Callable[[Image, np.random.Generator], Image]] = {}

@register("clean")
def clean(mask, rng): ...

@register("neon_cyan")
def neon_cyan(mask, rng): ...
```

v1 등록된 effect 19종:

| 카테고리 | 이름 |
|---------|-----|
| 채우기 | clean, solid_inverse, bold |
| 외곽선 | outline_thin, outline_thick, double_outline |
| 그림자 | drop_shadow, long_shadow |
| 채움 효과 | gradient_warm, gradient_cool, neon_cyan, neon_pink, stripe_30, stripe_90 |
| 배경 | noisy_bg |
| 기하 | rotated_italic, rotated_ccw, warp_a, warp_b |

**"clean"은 특권적 개념이 아니라 19개 중 하나일 뿐이다.** 이것이 v1 리팩터의
핵심 변화다.

기하 effect 4종(rotated_italic / rotated_ccw / warp_a / warp_b)은 사실상
글자 자체의 image-space augmentation이고, 프로토타입 패리티를 위해
effect에 묶어뒀다. 추후 augment 쪽으로 이동 가능.

### 3. `generate.py` — 단일 진입점

```
python generate.py <char>
  [--sources font:all | font:malgun | font:malgun,batang]
  [--effects all | clean | clean,neon_cyan,drop_shadow]
  [--out <path>]
  [--seed <int>]
```

기본값(`--sources font:all --effects all`)으로 실행하면
`sources × effects` 전체 matrix를 생성. `46 × 19 = 874` 개 PNG가 `out/U+9451/`에
찍힌다.

이전 프로토타입 동작의 재현:

| 구 동작 | 신 명령 |
|--------|--------|
| `render_systemfonts.py 鑑` | `generate.py 鑑 --effects clean` |
| `render_stylized.py 鑑`    | `generate.py 鑑 --sources font:malgun --effects all` |

### 4. `augment.py` — 실세계 조건 시뮬 (미결합)

5그룹 × 24 op 라이브러리. base_source와 effect_stack에 이어 세 번째 축이
되어야 하지만 **v1에서는 아직 generate.py CLI와 결합되지 않았다**.
결합되면 파이프라인은:

```
base_source → mask → effect_stack → RGB → augment_pipeline → RGB (다양화) → PNG
```

augment는 별도 CLI 없이 라이브러리 레벨로만 동작 가능. 예시 코드:

```python
from augment import apply_pipeline
import numpy as np

pipeline = [
    {"op": "rotate",         "angle":  [-8, 8]},
    {"op": "gaussian_blur",  "sigma":  [0.3, 1.5]},
    {"op": "gaussian_noise", "std":    [3, 15]},
    {"op": "jpeg",           "quality":[40, 90]},
]
rng = np.random.default_rng(0)
out = apply_pipeline(img, pipeline, rng)
```

### 5. `_legacy/` — 이전 프로토타입

`render_systemfonts.py`, `render_stylized.py`는 `scripts/_legacy/`로 이동.
README.md 포함. 코드 아카이브 용도로만 남김.

## v1 Variation Surface

### 실제로 현재 나오는 양

`generate.py 鑑` 기본 실행 결과: **874장 PNG** (46 sources × 19 effects).
이전 프로토타입은 65장(46 + 19, addition)이었으므로, **통합 한 번으로
곱으로 확장** 되었다.

### 축별 cardinality

| 축 | v1 cardinality | 설명 |
|---|----------------|-----|
| base_source | 46 (鑑 기준) | CJK 폰트 face. 국가/서체 계열 variation |
| effect_stack | 19 | 채움/외곽선/그림자/그라디언트/네온/줄무늬/노이즈/기하 |
| augment | ∞ (continuous) | 24 op × 연속 파라미터 조합. CLI 미결합 |

### 무엇이 variation으로 나오는가

- **base_source 축**: 한국 고딕/명조, 일본 명조/고딕, 중국 간체/번체 등
  동일 codepoint의 국가/서체 계열 차이. 곧 실세계에 존재하는 **자연스러운
  글리프 분포**.
- **effect_stack 축**: 포스터·간판·네온·인쇄·고저대비 등 **의도적 스타일링**.
- **augment 축 (결합 시)**: 블러·노이즈·JPEG·저조도·스캔·카메라 결함 등
  **찍힘/전송 후 열화**.

세 축은 **서로 직교**하며 곱으로 결합.

## v1 시점에 못 만드는 변형 축

구조적으로 v1 엔진이 **아직 생성할 수 없는** variation.

1. **효과 스택 중첩(layer grammar)**
   - 현 effect 하나 = 한 번의 full render. "외곽선 + 그라디언트 + 하이라이트 +
     그림자"를 한 이미지에 쌓을 수 없음.
   - 결과: "꾸며진 글자"(간판·포스터·게임 UI)의 시각 복잡도 못 닿음.

2. **scene 배경 합성**
   - 배경이 solid / noise / stripe pattern 뿐.
   - 실제 글자는 사진·일러스트·레이아웃 위에 얹혀 있음. scene 축이 0.

3. **glyph 자체 비선형 변형**
   - rotate / shear / perspective는 **이미지 전체**에 걸림.
   - 글자 **내부** 변형(세로 압축, 획 굵기 불균형, rounded corner,
     baseline 곡률, 획 끝 장식)은 불가.
   - "있는 폰트만 쓴다"는 한계.

4. **재질 시뮬 (material)**
   - 메탈릭, 크롬, 우드, 종이 위 잉크, 붓질 등.

5. **조명 시뮬 (lighting)**
   - bevel / emboss / 방향성 하이라이트 부재.

6. **해상도·크기 다양성**
   - 출력이 항상 256×256 고정. 실제는 훨씬 작은 글자일 수 있음.

7. **non-font base source**
   - `SvgStrokeSource`, `ProceduralSource`, `HandwritingSource` 미구현.
   - 결과: 폰트 커버리지 없는 tail codepoint 진입 경로 없음.

8. **augment 결합**
   - 라이브러리는 있지만 generate.py에 연결되지 않아 실산출물로 못 나옴.

## 다음 반복에서 열어야 할 축

일정 고려 없이 **variation surface 확장량**만 기준으로 우선순위.

1. **augment 결합** — 라이브러리는 이미 있음. CLI 결합만으로 변형 공간이
   addition에서 multiplication으로 두 번째 곱 확장. 비용 대비 효과 최대.
2. **Effect stacking grammar** — 단일 effect가 아닌 layer list로 전환.
   스타일 공간이 이산 19에서 조합 무한으로.
3. **Scene compositor** — 배경 풀 + alpha blend. "배경" 축이 0에서 N으로.
4. **SvgStrokeSource** — base_source 축에 새 값 추가. 폰트 없는 tail
   codepoint 진입 가능.
5. **Glyph self-deform** — 폰트 mask 단에서 비선형 변형. 서체 디자인 레벨
   variation 확보.
6. **Material / lighting layers** — effect stacking이 완성된 후 추가.

각 항목은 독립이고 어느 순서로 해도 전체 surface가 확장됨.

## 결론

v1의 본질적 변화는 코드량이 아니라 **멘털 모델의 통합**이다.

- 구: "fonts"와 "stylized"라는 두 평행 파이프라인.
- 신: 하나의 파이프라인, 세 pluggable 축 (source / effect / augment).
  clean·font·기타 모든 경우가 이 세 축 위의 특수 케이스.

실행면에서 v1은 `generate.py <char>`만으로 CJK 전체 × 19 스타일 매트릭스를
한 번에 찍는다. 향후 augment CLI 결합, 추가 base_source 종류, layer grammar
등 어느 축을 확장해도 기존 구조를 **깨지 않고** 덧붙일 수 있다.
