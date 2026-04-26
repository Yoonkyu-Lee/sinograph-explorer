# KanjiVG Manual

이 문서는 [KanjiVG](./KanjiVG) 원본 SVG와 이 프로젝트에서 추가로 만든 [strokes_kanjivg.jsonl](./KanjiVG/strokes_kanjivg.jsonl)의 구조를 정리한 매뉴얼이다.

KanjiVG는 `Unihan`이나 `e-hanja`처럼 뜻, 음, 변이 관계를 설명하는 사전 DB가 아니다. 성격을 한 문장으로 줄이면 **일본식 표준 자형을 기준으로 한 획순 SVG와 획 메타데이터 DB**다. 특히 이 프로젝트에서는 원본 SVG의 획 centerline을 파싱해서, 합성 OCR 엔진이 바로 쓸 수 있는 median stroke JSONL로 정규화해 두었다.

## Source And Licensing

- **프로젝트**: KanjiVG: Kanji Vector Graphics
- **원본 문서**: https://kanjivg.tagaini.net/
- **로컬 보관 위치**: [KanjiVG](./KanjiVG)
- **라이선스**: Creative Commons Attribution-Share Alike 3.0
- **라이선스 파일**: [COPYING](./KanjiVG/COPYING)

### License Caveat

KanjiVG는 CC BY-SA 3.0이다. 즉 단순 내부 실험이나 분석에는 다루기 쉽지만, 이 데이터를 변형한 결과물을 외부 배포하거나 앱에 탑재할 때는 attribution과 share-alike 조건을 확인해야 한다.

특히 이 프로젝트의 [strokes_kanjivg.jsonl](./KanjiVG/strokes_kanjivg.jsonl)은 원본 SVG path를 파싱하고 좌표계를 바꾼 파생 데이터다. 원본의 라이선스 성격이 사라지는 것이 아니므로, 배포 단계에서는 KanjiVG 출처와 라이선스 표기를 유지하는 것이 안전하다.

## File Inventory

현재 로컬 [KanjiVG](./KanjiVG) 디렉터리의 핵심 파일은 다음과 같다.

- [kanji/](./KanjiVG/kanji)
  - 원본 SVG 파일 디렉터리
  - `*.svg` 총 **11,662개**
  - base 파일 **6,703개**
  - variant 파일 **4,959개**
- [strokes_kanjivg.jsonl](./KanjiVG/strokes_kanjivg.jsonl)
  - 이 프로젝트에서 생성한 median stroke JSONL
  - **6,446 records**
  - **79,246 strokes**
- [extract_anomalies.jsonl](./KanjiVG/extract_anomalies.jsonl)
  - 추출 중 이상 케이스 기록
  - 현재 파일 크기 0, 즉 기록된 anomaly 없음
- [kvg-index.json](./KanjiVG/kvg-index.json)
  - KanjiVG upstream viewer용 문자별 SVG 파일 index
- [README.md](./KanjiVG/README.md)
  - upstream 파일 구성과 릴리스 설명
- [kanjivg.py](./KanjiVG/kanjivg.py), [kvg.py](./KanjiVG/kvg.py), [xmlhandler.py](./KanjiVG/xmlhandler.py)
  - upstream Python helper code

추출 스크립트는 [../db_mining/kanjivg_extract/scripts/extract_kanjivg_medians.py](../db_mining/kanjivg_extract/scripts/extract_kanjivg_medians.py)에 있다.

## High-Level Positioning

KanjiVG는 Sinograph Explorer 전체 스택 안에서 다음 위치에 놓는 것이 자연스럽다.

- **not** a semantic dictionary
- **not** a Korean hun-eum source
- **not** a variant-family authority
- **is** a stroke-order source
- **is** a stroke centerline source
- **is** a useful graphics source for synthetic OCR data

역할 분담으로 보면:

- Unihan = codepoint, variant, 다국가 표준 backbone
- e-hanja = 한국어 뜻/훈음/한국식 사전 정보
- KANJIDIC2 = 일본어 reading, 사전 index
- MakeMeAHanzi = 중국계 decomposition + graphics
- **KanjiVG = 일본식 glyph의 획순, 획종, centerline**

즉 KanjiVG는 "이 글자가 무슨 뜻인가"보다 "이 글자를 어떤 획 순서와 선형 궤적으로 그리는가"에 강하다.

## Original SVG Structure

원본 SVG는 문자 하나를 하나의 SVG 파일로 둔다. 일반 base 파일명은 Unicode codepoint를 5자리 hex로 zero-pad한 형태다.

예:

- `04e00.svg` = U+4E00 `一`
- `05b66.svg` = U+5B66 `学`
- `09f8d.svg` = U+9F8D `龍`

variant 파일은 같은 base stem 뒤에 tag가 붙는다.

예:

- `05b70.svg`
- `05b70-Kaisho.svg`
- `05b43-KaishoVtLst.svg`

이 프로젝트의 median 추출은 **variant 파일을 스킵하고 base 파일만 처리**한다. 한 글자에 여러 서체/변형을 모두 넣으면 class 중복이 생기고, downstream에서 "한 codepoint당 하나의 canonical stroke source"라는 가정이 깨지기 때문이다.

### SVG 내부 구조

대표적으로 `04e00.svg`는 다음 같은 구조를 가진다.

```xml
<svg width="109" height="109" viewBox="0 0 109 109">
  <g id="kvg:StrokePaths_04e00"
     style="fill:none;stroke:#000000;stroke-width:3;stroke-linecap:round;stroke-linejoin:round;">
    <g id="kvg:04e00" kvg:element="一" kvg:radical="general">
      <path id="kvg:04e00-s1" kvg:type="㇐" d="M11,54.25c3.19,..."/>
    </g>
  </g>
  <g id="kvg:StrokeNumbers_04e00">...</g>
</svg>
```

중요한 부분은 세 가지다.

- `<g id="kvg:{cp}" kvg:element="...">`
  - 문자 자체를 나타낸다.
  - 추출 스크립트는 이 값을 `char`로 우선 사용하고, 없으면 `chr(cp)`로 fallback한다.
- `<path id="kvg:{cp}-s{N}">`
  - `N`이 획 순서다.
  - `s1`, `s2`, ... 순으로 첫 획부터 마지막 획까지 대응된다.
- `kvg:type`
  - 획종 정보다.
  - 예: `㇐`, `㇑`, `㇒`, `㇔`, `㇏`, `㇕a`
  - 현재 JSONL에서는 `kind`로 보존한다.

KanjiVG의 `<path d="...">`는 닫힌 outline이 아니라 **stroke centerline**이다. SVG style의 `stroke-width:3`으로 굵기를 입혀 렌더링하는 방식이다. 이 점이 e-hanja online과 가장 큰 차이다.

## `strokes_kanjivg.jsonl`

### File Shape

[strokes_kanjivg.jsonl](./KanjiVG/strokes_kanjivg.jsonl)은 one JSON object per line 형식이다. 실질적인 primary key는 `char` 또는 `cp`다.

예:

```json
{
  "char": "一",
  "cp": 19968,
  "viewbox": [109, 109],
  "y_pivot": 109,
  "strokes": [
    {
      "order": 1,
      "kind": "㇐",
      "median": [[11.0, 54.75], [24.0, 54.0], "..."],
      "width": 3.0
    }
  ]
}
```

### Observed Top-Level Fields

- `char`
- `cp`
- `viewbox`
- `y_pivot`
- `strokes`

현재 로컬 파일 기준:

- records: **6,446**
- total strokes: **79,246**
- strokes per char:
  - min: **1**
  - median: **12**
  - mean: **12.29**
  - max: **30**
- viewbox:
  - `[109, 109]`: **6,446 / 6,446**

### Field Descriptions

#### `char`

- 해당 codepoint의 문자
- 다른 DB와 join할 때 가장 읽기 쉬운 natural key
- 내부적으로는 `cp`와 항상 같이 보관하는 것이 안전하다

#### `cp`

- Unicode codepoint integer
- 파일명, Unihan `U+XXXX`, e-hanja online `hex`와 연결할 때 가장 안정적인 key

권장 join:

```python
kanjivg_key = record["cp"]
unihan_key = f"U+{record['cp']:04X}"
```

#### `viewbox`

- 항상 `[109, 109]`
- 원본 KanjiVG SVG의 `viewBox="0 0 109 109"`에서 온 값
- MakeMeAHanzi나 e-hanja online의 1024 계열 좌표계와 직접 같은 scale이 아니다

#### `y_pivot`

- 항상 `109`
- 원본 SVG는 y-down 좌표계다.
- 추출 결과는 합성 엔진에서 쓰기 쉽도록 y-up 좌표계로 뒤집었다.

변환 개념:

```text
y_math = 109 - y_svg
```

즉 JSONL의 `median` 좌표는 이미 y-up이다. 다시 SVG 원본 좌표로 되돌릴 때만 `109 - y`를 적용하면 된다.

#### `strokes`

- 획 배열
- `order` 오름차순으로 정렬되어 있다
- 각 원소는 `order`, `kind`, `median`, `width`를 가진다

### Stroke Fields

#### `order`

- 1부터 시작하는 획 순서
- 원본 path id의 `s{N}`에서 추출
- downstream에서 stroke-order animation, per-stroke perturbation, 획 dropout 등을 적용할 때 기준이 된다

#### `kind`

- KanjiVG `kvg:type`
- 획종을 나타내는 문자열
- 현재 로컬 추출 결과에서 distinct kind는 **91종**

상위 빈도 예:

- `㇐`: 13,381
- `㇒`: 11,980
- `㇑`: 10,130
- `㇔`: 8,344
- `㇐a`: 5,612
- `㇑a`: 5,490

이 값은 stroke morphology를 추정하는 힌트로는 유용하지만, 국가별 표준 획종 체계의 절대 기준으로 보면 안 된다. KanjiVG 내부 annotation 체계의 값으로 보는 것이 안전하다.

#### `median`

- 획의 centerline polyline
- 좌표는 `[x, y]` float pair list
- 현재 추출 스크립트는 각 획을 최대 7개 waypoint로 downsample한다.
- 현재 로컬 결과에서는 모든 stroke가 7개 waypoint를 가진다.

중요한 점:

- 이것은 outline이 아니다.
- 이미 획 중심선이다.
- e-hanja online의 `strokes_animated.jsonl`처럼 닫힌 polygon path를 기대하면 안 된다.

#### `width`

- 항상 `3.0`
- 원본 SVG의 CSS `stroke-width:3`을 그대로 반영한 값
- 실제 렌더링에서 더 두껍게 그리고 싶으면 loader나 rasterizer 쪽의 `width_scale`로 조절하는 것이 맞다.

## Coverage

### File-Level Coverage

원본 [kanji/](./KanjiVG/kanji) 디렉터리:

| 항목 | 개수 |
|---|---:|
| 전체 SVG | 11,662 |
| base SVG | 6,703 |
| variant SVG | 4,959 |
| `strokes_kanjivg.jsonl` records | 6,446 |

`base SVG 6,703`과 `JSONL 6,446`의 차이는 추출 스크립트가 비한자 항목을 필터링하기 때문이다. KanjiVG는 kana, ASCII, punctuation, fullwidth symbol 같은 stroke-order 자료도 함께 가진다. OCR 학습에서 CJK ideograph class가 아닌 항목은 noise가 되므로 제외했다.

### Unicode Block Coverage

[strokes_kanjivg.jsonl](./KanjiVG/strokes_kanjivg.jsonl) 기준:

| Block | Records |
|---|---:|
| CJK Unified Ideographs | 6,413 |
| CJK Compatibility Ideographs | 9 |
| CJK Radicals Supplement | 15 |
| CJK Extension A | 3 |
| CJK Extension B | 6 |

즉 KanjiVG는 거의 전부 BMP의 일반 CJK Unified 영역에 몰려 있다. Ext A/B나 compatibility 영역은 매우 얇다. 희귀 한자 tail coverage를 기대하는 DB가 아니라, 비교적 상용 일본 한자의 고품질 획순 데이터로 보는 것이 맞다.

## Extraction Pipeline

추출 스크립트:

[../db_mining/kanjivg_extract/scripts/extract_kanjivg_medians.py](../db_mining/kanjivg_extract/scripts/extract_kanjivg_medians.py)

핵심 처리 단계:

1. `db_src/KanjiVG/kanji/*.svg` 스캔
2. filename stem에 `-`가 있으면 variant로 보고 스킵
3. 5자리 hex stem을 codepoint로 파싱
4. CJK ideograph/radical block whitelist를 통과하지 못하면 스킵
5. `<path id="kvg:{cp}-s{N}">`만 추출
6. SVG path의 curve를 polyline으로 flatten
7. 최대 7개 waypoint로 downsample
8. y-down 좌표를 y-up 좌표로 변환
9. `strokes_kanjivg.jsonl`에 기록

이 파이프라인은 skeletonization을 하지 않는다. KanjiVG path 자체가 centerline이기 때문이다. e-hanja online처럼 outline polygon에서 중심선을 복원하는 경우와 비교하면 훨씬 정보 손실이 적고 안정적이다.

## Comparison With Other Stroke Sources

### KanjiVG vs MakeMeAHanzi

공통점:

- 둘 다 stroke-level graphics를 제공한다.
- 둘 다 median 기반 합성 엔진에 넣기 쉽다.
- 둘 다 일반 사전 DB가 아니다.

차이점:

| 항목 | KanjiVG | MakeMeAHanzi |
|---|---|---|
| 주된 glyph 기준 | 일본식 Kanji | 중국계 Hanzi |
| records | 6,446 | 9,574 |
| 좌표계 | 109 x 109 | 1024 계열 |
| centerline | 원본 SVG path 자체 | `medians` 필드 |
| stroke kind | `kvg:type` 보유 | 명시적 획종 약함 |
| decomposition | 강하지 않음 | 강함 |

KanjiVG는 획종 annotation이 풍부하고 stroke order가 명확하다. 반면 coverage는 MakeMeAHanzi보다 좁고, decomposition/etymology 쪽 정보는 MakeMeAHanzi가 더 직접적이다.

### KanjiVG vs e-hanja online

| 항목 | KanjiVG | e-hanja online |
|---|---|---|
| 성격 | 일본식 stroke-order SVG | 한국 e-hanja 웹 SVG |
| 처리 가능 records | 6,446 | 16,329 animated |
| 전체 SVG records | 11,662 원본 SVG | 76,013 clean SVG |
| stroke geometry | centerline path | outline path + medianized 파생 |
| 좌표계 | 109 x 109 | 1024 x 1152 |
| 희귀 한자 coverage | 약함 | Ext A/B 일부 강함 |
| 한국식 자형성 | 약함 | 강함 |

KanjiVG는 작지만 깨끗한 centerline source이고, e-hanja online은 넓지만 outline 기반 후처리가 필요한 source다. OCR 합성에서는 둘을 경쟁 관계로 보기보다 서로 다른 style/domain source로 보는 것이 맞다.

## Recommended Usage

### 1. Synthetic OCR Source

가장 직접적인 용도는 stroke-based synthetic rendering이다.

권장 방식:

```python
for line in open("db_src/KanjiVG/strokes_kanjivg.jsonl", encoding="utf-8"):
    rec = json.loads(line)
    char = rec["char"]
    strokes = rec["strokes"]
```

렌더링할 때는:

- `viewbox = [109, 109]`
- `y_pivot = 109`
- `median`은 이미 y-up
- `width`는 기본 3.0, 필요시 scale

### 2. Stroke Order / Stroke Count Reference

KanjiVG의 `order`는 명시적이고 안정적이다. 다음 작업에 유용하다.

- 획순 animation
- stroke count sanity check
- 한 글자의 stroke-level augmentation
- stroke omission/noise robustness test

단, 일본식 표준 획순이므로 한국 교육 표준이나 중국 표준과 다를 수 있다.

### 3. Glyph Style Diversity

e-hanja online이나 MakeMeAHanzi만 쓰면 glyph style이 한쪽으로 치우칠 수 있다. KanjiVG를 추가하면 일본식 자형 variation을 합성 데이터에 섞을 수 있다.

다만 OCR 목표가 "한국 표준 자형"이면 KanjiVG를 주 source로 쓰기보다 보조 source로 쓰는 편이 안전하다.

## Caveats

### 1. Japanese Glyph Bias

KanjiVG는 일본 Kanji 사용 환경을 강하게 반영한다. 같은 Unicode codepoint라도 한국, 중국, 일본에서 glyph shape가 다를 수 있다.

예를 들어 같은 codepoint가 폰트와 지역 표준에 따라 점, 삐침, 구성요소 위치가 달라질 수 있다. KanjiVG는 이 차이를 모두 포괄하는 pan-CJK authority가 아니다.

### 2. Coverage Is Narrow

`strokes_kanjivg.jsonl`은 6,446자다. Unihan이나 e-hanja online처럼 수만 자 coverage를 기대하면 안 된다.

특히:

- Ext A/B tail
- compatibility ideograph
- 한국 고유 용례
- 희귀 인명용 한자

이런 영역은 e-hanja online, Unihan, CNS11643 쪽이 더 중요하다.

### 3. Variants Are Currently Skipped

원본에는 `-Kaisho`, `-TenLst`, `-VtLst` 같은 variant SVG가 많지만 현재 추출 결과에는 들어가지 않는다.

이 결정은 downstream simplicity를 위한 것이다. 나중에 style augmentation source로 variant까지 쓰고 싶다면 다음이 필요하다.

- base와 variant를 같은 `cp` 아래 multiple glyph forms로 모델링
- source/style tag 추가
- train split에서 같은 codepoint의 variant leakage 관리

### 4. Width Is Not Physical Stroke Width

`width = 3.0`은 원본 SVG의 display stroke width다. 붓글씨나 인쇄체의 실제 획 두께를 추정한 값이 아니다.

따라서 width jitter를 줄 때는 "원본 선 굵기에서 자연스럽게 변형"이라기보다 "centerline을 렌더링하는 그래픽 파라미터"로 해석해야 한다.

## Practical Lookup Recipes

### 문자에서 원본 SVG 찾기

```python
char = "一"
cp = ord(char)
path = f"db_src/KanjiVG/kanji/{cp:05x}.svg"
```

### 문자에서 JSONL record 찾기

파일이 작으므로 실험 단계에서는 dict로 올려도 된다.

```python
records = {}
with open("db_src/KanjiVG/strokes_kanjivg.jsonl", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        records[rec["char"]] = rec
```

### 다른 DB와 join하기

권장 join key 우선순위:

1. `cp` integer
2. `char`
3. filename stem

`cp`가 가장 안전하다. `char`는 사람이 읽기 좋지만, surrogate pair나 normalization 이슈를 피하려면 내부 join은 codepoint integer로 두는 편이 좋다.

## Summary

KanjiVG는 **작지만 깨끗한 stroke centerline DB**다.

핵심만 정리하면:

- 원본 SVG: 11,662개, 그중 base 6,703개
- 이 프로젝트 추출 결과: 6,446자, 79,246획
- 좌표계: 109 x 109, y-up으로 변환 완료
- geometry: outline이 아니라 median centerline
- 강점: 획순, 획종, 깨끗한 centerline
- 약점: 일본식 glyph bias, 좁은 coverage, 의미/훈음 정보 없음

따라서 Sinograph Explorer에서는 KanjiVG를 canonical dictionary가 아니라 **stroke rendering source이자 style diversity source**로 쓰는 것이 가장 적합하다.
