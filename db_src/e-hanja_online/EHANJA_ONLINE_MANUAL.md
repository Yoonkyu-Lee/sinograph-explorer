# e-hanja Online Manual

이 문서는 온라인 e-hanja 수집 결과인 [e-hanja_online](./e-hanja_online) 디렉터리의 구조와 활용법을 정리한 매뉴얼이다.

중요한 구분부터 해야 한다. [e-hanja](./e-hanja)는 모바일 앱에서 추출한 SQLite/CSV 계열 사전 DB이고, [e-hanja_online](./e-hanja_online)은 `www.e-hanja.kr`, `img.e-hanja.kr`, `tool.img.e-hanja.kr`에서 별도로 수집한 온라인 데이터다. 이름은 비슷하지만 수집 경로, coverage, 데이터 성격이 다르다.

현재 `db_src/e-hanja_online/`에 정리된 것은 주로 **SVG 기반 획 데이터**다. 온라인 API에서 받은 훈음/자해/변이 JSON과 상세 HTML 원본은 [../db_mining/RE_e-hanja_online/data](../db_mining/RE_e-hanja_online/data)에 수집 원본으로 남아 있다.

## Source And Licensing

- **원천 사이트**: http://www.e-hanja.kr/
- **SVG 서버**: `http://img.e-hanja.kr/hanjaSvg/aniSVG/{folder}/{cp}.svg`
- **JSON API 서버**: `http://tool.img.e-hanja.kr/hanjaSvg/asp/dbHandle.comsTree.asp`
- **상세 HTML 서버**: `http://www.e-hanja.kr/dic/contents/jajun_contentA.asp`
- **로컬 가공 결과**: [e-hanja_online](./e-hanja_online)
- **수집 작업 공간**: [../db_mining/RE_e-hanja_online](../db_mining/RE_e-hanja_online)

### License Caveat

이 데이터는 명시적 오픈 라이선스가 확인된 공개 dataset이라기보다, 온라인 사전 사이트를 정찰하고 수집한 자료다. 수집 계획 문서도 이 데이터를 **프로젝트 내부 학습/분석용**으로 쓰는 것을 전제로 한다.

따라서:

- 외부 재배포 금지로 보는 것이 안전하다.
- 앱에 raw SVG나 raw HTML을 직접 탑재하는 것은 별도 검토가 필요하다.
- 학습 데이터 생성에 쓰더라도 provenance를 분리해서 남겨야 한다.
- 사이트에 부하를 주는 재수집은 rate limit/backoff를 지켜야 한다.

이 매뉴얼은 로컬 프로젝트 안에서 구조를 이해하고 downstream 처리를 하기 위한 문서다.

## Relationship To Mining Workspace

수집과 가공은 다음 흐름으로 이루어졌다.

```text
db_mining/RE_e-hanja_online/
  data/svg/{HEX}.svg       raw crawled SVG
  data/tree/{HEX}.json     online JSON API 3종 통합
  data/detail/{HEX}.html   상세 HTML 원본
        |
        | classify / extract / medianize / watermark strip
        v
db_src/e-hanja_online/
  svg/{HEX}.svg
  strokes_manifest.jsonl
  strokes_animated.jsonl
  strokes_medianized.jsonl
  extract_anomalies.jsonl
  medianize_anomalies.jsonl
```

관련 문서:

- [../db_mining/RE_e-hanja_online/SITE_ANALYSIS.md](../db_mining/RE_e-hanja_online/SITE_ANALYSIS.md)
- [../db_mining/RE_e-hanja_online/COLLECTION_PLAN.md](../db_mining/RE_e-hanja_online/COLLECTION_PLAN.md)
- [../db_mining/RE_e-hanja_online/PROCESSING_PLAN.md](../db_mining/RE_e-hanja_online/PROCESSING_PLAN.md)
- [../db_mining/RE_e-hanja_online/README.md](../db_mining/RE_e-hanja_online/README.md)

## File Inventory

현재 [e-hanja_online](./e-hanja_online)에는 다음 파일들이 있다.

- [svg/](./e-hanja_online/svg)
  - watermark를 제거한 clean SVG
  - **76,013 files**
  - animated와 static SVG가 섞여 있다
- [strokes_manifest.jsonl](./e-hanja_online/strokes_manifest.jsonl)
  - 모든 clean SVG의 분류 manifest
  - **76,013 records**
- [strokes_animated.jsonl](./e-hanja_online/strokes_animated.jsonl)
  - animated SVG에서 획별 outline path를 추출한 파일
  - **16,329 records**
  - **210,158 outline strokes**
- [strokes_medianized.jsonl](./e-hanja_online/strokes_medianized.jsonl)
  - outline stroke를 skeletonize해서 median + width 형태로 바꾼 파일
  - **16,329 records**
  - **210,156 median strokes**
- [extract_anomalies.jsonl](./e-hanja_online/extract_anomalies.jsonl)
  - outline 추출 단계의 의심 케이스
  - 현재 **26 records**
- [medianize_anomalies.jsonl](./e-hanja_online/medianize_anomalies.jsonl)
  - medianization 단계의 의심 케이스
  - 현재 **2 records**

수집 원본 쪽 규모:

| 위치 | 내용 | 현재 파일 수 |
|---|---|---:|
| `db_mining/RE_e-hanja_online/data/svg/*.svg` | raw SVG | 76,013 |
| `db_mining/RE_e-hanja_online/data/svg/*.404` | SVG missing marker | 83 |
| `db_mining/RE_e-hanja_online/data/tree/*.json` | JSON API 결과 | 76,013 |
| `db_mining/RE_e-hanja_online/data/detail/*.html` | 상세 HTML | 75,669 |

초기 정찰 문서에는 예상 coverage가 71,716자로 적혀 있는 부분이 있지만, 이 매뉴얼에서는 **현재 로컬 산출물 기준 76,013 SVG**를 기준으로 설명한다.

## High-Level Positioning

e-hanja online은 이 프로젝트에서 가장 중요한 stroke coverage 확장 source 중 하나다.

- **not** the same DB as mobile `e-hanja`
- **not** a clean open-license public dataset
- **is** a large Korean-facing online hanja source
- **is** a canonical-ish Korean glyph SVG source
- **is** a high-coverage stroke source for characters not covered by MakeMeAHanzi
- **is** a useful bridge between Korean dictionary data and OCR synthesis

역할 분담:

- mobile e-hanja = 한국어 뜻/훈음/관계형 사전 테이블
- **e-hanja online = 더 넓은 온라인 coverage + SVG 획 자원**
- MakeMeAHanzi = compact Chinese decomposition + stroke graphics
- KanjiVG = Japanese stroke centerline
- Unihan = Unicode-level codepoint/variant backbone

e-hanja online의 가장 큰 가치는 **한국 쪽 자형으로 된 SVG를 넓게 확보했다는 점**이다.

## Source Axes

온라인 수집은 원래 3축으로 설계되었다.

### 1. SVG Axis

저장 위치:

- raw: `db_mining/RE_e-hanja_online/data/svg/{HEX}.svg`
- clean: [e-hanja_online/svg/{HEX}.svg](./e-hanja_online/svg)

엔드포인트:

```text
http://img.e-hanja.kr/hanjaSvg/aniSVG/{(cp & 0xFF00):X}/{cp:X}.svg
```

이 축이 현재 `db_src/e-hanja_online`의 중심이다.

### 2. Tree JSON Axis

저장 위치:

- `db_mining/RE_e-hanja_online/data/tree/{HEX}.json`

엔드포인트:

```text
POST http://tool.img.e-hanja.kr/hanjaSvg/asp/dbHandle.comsTree.asp
```

한 codepoint에 대해 세 request를 합쳐 저장한다.

- `getHunum`
  - 훈음
  - 예: `{hanja: "一", hRead: "한 일"}`
- `getJahae`
  - 뜻 목록
  - 예: `{hanja, orderA, orderB, meaning, root_snd}`
- `getSchoolCom`
  - 변이 관계
  - `hDup`, `bonja`, `sokja`, `yakja`, `goja`, `waja`, `simple`, `kanji`, `dongja`, `tongja`

예를 들어 U+4E00 `一`의 tree JSON은 `한 일`, 여러 뜻 목록, `goja=弌`, `dongja=𠤪,壹` 같은 정보를 담는다.

### 3. Detail HTML Axis

저장 위치:

- `db_mining/RE_e-hanja_online/data/detail/{HEX}.html`

엔드포인트:

```text
POST http://www.e-hanja.kr/dic/contents/jajun_contentA.asp
```

상세 HTML은 14개 정보 블록을 포함한다.

- 훈음과 등급
- 자해 뜻 목록
- 부수와 어원
- 총획수와 부수 획수
- 모양자/decomposition
- 자원/etymology 설명
- 영어 gloss
- 한어병음
- 교육용 분류
- 한자검정 급수
- 대법원 인명용 여부
- 기타 분류
- 동자 thumbnail
- 유의/관련 복합어

현재 매뉴얼의 주 대상은 `db_src/e-hanja_online`의 SVG 가공 결과지만, 나중에 canonical DB를 확장할 때 tree/detail 원본도 중요한 보강 source가 된다.

## Two SVG Types

e-hanja online SVG는 크게 두 종류다.

| 타입 | 판별 | 구조 | 획 분리 | 현재 처리 |
|---|---|---|---|---|
| `animated` | `class="ani-svg"` | 획마다 `<path class="stroke-normal|stroke-radical">` | 가능 | outline, median 모두 가공 |
| `static` | `class="svg"` | 글자 전체가 단일 `<path class="path-normal">` | 불가능 | manifest와 clean SVG만 보존 |

이 차이가 매우 중요하다.

animated SVG는 획 하나하나가 닫힌 outline path로 분리되어 있다. 예를 들어 `一`은 다음처럼 한 획 path가 `U4E00d1`로 들어 있다.

```xml
<svg id="U4E00ani" class="ani-svg" viewBox="0 0 1024 1152">
  <g transform="scale(1,-1) translate(0, -871)" fill="currentColor">
    <path id="U4E00d1"
          d="M56 312Q76 326..."
          class="stroke-radical"/>
  </g>
</svg>
```

반대로 static SVG는 글자 전체가 하나의 monolithic path라서 "어느 부분이 몇 번째 획인가"를 안정적으로 복구할 수 없다. skeletonization으로 추정할 수는 있지만, 획 순서와 획 경계를 잃은 상태라 현재 파이프라인에서는 보류한다.

## `strokes_manifest.jsonl`

### Role

[strokes_manifest.jsonl](./e-hanja_online/strokes_manifest.jsonl)은 모든 clean SVG에 대해 "이 SVG가 animated인지 static인지"를 기록하는 index다. downstream은 이 파일을 먼저 보고 처리 가능한 subset을 결정한다.

### Schema

```json
{
  "cp": 11904,
  "hex": "2E80",
  "char": "⺀",
  "type": "animated",
  "stroke_count": 2,
  "viewbox": [1024, 1152]
}
```

필드:

- `cp`
  - Unicode codepoint integer
- `hex`
  - uppercase hex string, 파일명 stem과 대응
- `char`
  - 문자 자체
- `type`
  - `animated` 또는 `static`
- `stroke_count`
  - animated이면 획 수 integer
  - static이면 `null`
- `viewbox`
  - SVG viewBox width/height

### Observed Counts

현재 로컬 파일 기준:

| type | records |
|---|---:|
| animated | 16,329 |
| static | 59,684 |
| total | 76,013 |

animated stroke count:

- min: **1**
- median: **13**
- mean: **12.87**
- max: **39**

viewBox:

| viewbox | records |
|---|---:|
| `[1024, 1152]` | 75,957 |
| `[1000, 1152]` | 56 |

단, animated 16,329자는 모두 `[1024, 1152]`이다. `[1000, 1152]` 케이스 56개는 모두 static 쪽이다.

### Block Coverage

manifest 전체:

| Block | animated | static |
|---|---:|---:|
| Radicals Supplement | 115 | 0 |
| Kangxi Radicals | 214 | 0 |
| CJK Unified | 13,669 | 7,320 |
| CJK Ext A | 1,018 | 5,574 |
| CJK Compat | 327 | 145 |
| CJK Ext B | 907 | 41,811 |
| CJK Ext C | 0 | 4,149 |
| CJK Ext D | 0 | 222 |
| Other | 79 | 463 |

해석:

- BMP CJK Unified와 일부 Ext A/B는 animated로 획 분리 가능하다.
- Ext B/C/D tail 대부분은 static이다.
- 따라서 e-hanja online 전체 SVG coverage는 매우 넓지만, 현재 stroke-level synthesis에 바로 쓸 수 있는 coverage는 16,329자다.

## `strokes_animated.jsonl`

### Role

[strokes_animated.jsonl](./e-hanja_online/strokes_animated.jsonl)은 animated SVG에서 획별 outline path를 추출한 파일이다.

이 파일은 **lossless에 가까운 outline layer**다. 원본 SVG path의 `d` 문자열을 그대로 보존하므로, 나중에 outline renderer를 만들거나 medianization을 다시 할 때 기준 입력으로 쓸 수 있다.

### Schema

```json
{
  "cp": 11904,
  "hex": "2E80",
  "char": "⺀",
  "viewbox": [1024, 1152],
  "transform": "scale(1,-1) translate(0, -782)",
  "strokes": [
    {
      "order": 1,
      "kind": "radical",
      "d": "M631 376L642 358..."
    },
    {
      "order": 2,
      "kind": "radical",
      "d": "M636 34L652 8..."
    }
  ]
}
```

### Top-Level Fields

#### `cp`, `hex`, `char`

- 문자 identity
- join key로는 `cp`가 가장 안정적이다
- `hex`는 SVG filename과 직접 대응한다

#### `viewbox`

- animated는 모두 `[1024, 1152]`
- full glyph canvas 크기다

#### `transform`

- 원본 SVG의 outer `<g transform="...">`
- 예: `scale(1,-1) translate(0, -871)`
- e-hanja SVG는 y축을 뒤집고 baseline을 이동하는 transform을 쓴다.
- outline path를 직접 렌더링하려면 이 transform을 적용해야 한다.

#### `strokes`

- 획별 outline path 배열
- `order` 오름차순
- `d`는 SVG path string

### Stroke Fields

#### `order`

- 획 순서
- 원본 path id의 `U{CP}d{N}`에서 `N`을 추출한다.
- `d1`, `d2`, ... 가 animation 순서와 대응된다.

#### `kind`

- `normal` 또는 `radical`
- 원본 class에서 온 값:
  - `stroke-normal` -> `normal`
  - `stroke-radical` -> `radical`

현재 `strokes_animated.jsonl` 기준:

| kind | strokes |
|---|---:|
| normal | 130,198 |
| radical | 79,960 |
| total | 210,158 |

`radical`은 "이 획이 부수에 속한다"는 시각적 annotation으로 볼 수 있다. 학습 데이터 합성에서는 부수 강조, radical-aware augmentation, component-aware analysis에 쓸 수 있다.

#### `d`

- 닫힌 outline SVG path
- centerline이 아니다
- KanjiVG의 `median`이나 MakeMeAHanzi의 `medians`와 바로 같은 의미가 아니다

중요:

- e-hanja outline은 이미 채워 그릴 수 있는 polygon/Bezier path다.
- 선을 굵게 그리는 방식이 아니라, path 내부를 fill하는 방식으로 렌더링해야 한다.

## `strokes_medianized.jsonl`

### Role

[strokes_medianized.jsonl](./e-hanja_online/strokes_medianized.jsonl)은 e-hanja outline을 skeletonize해서 MakeMeAHanzi/KanjiVG와 비슷한 **median + width** 구조로 바꾼 파일이다.

목적은 기존 `svg_stroke` 계열 합성 엔진이 같은 loader 형태로 읽을 수 있게 하는 것이다.

### Schema

```json
{
  "char": "⺀",
  "cp": 11904,
  "viewbox": [1024, 1152],
  "y_pivot": 1152,
  "strokes": [
    {
      "order": 1,
      "kind": "radical",
      "median": [[362.0, 917.0], [413.4, 899.8], "..."],
      "width": 32.79
    }
  ]
}
```

### Observed Counts

현재 로컬 파일 기준:

- records: **16,329**
- median strokes: **210,156**
- strokes per char:
  - min: **1**
  - median: **13**
  - mean: **12.87**
  - max: **39**
- viewbox:
  - `[1024, 1152]`: **16,329 / 16,329**
- `y_pivot`:
  - `1152`

stroke kind:

| kind | strokes |
|---|---:|
| normal | 130,196 |
| radical | 79,960 |
| total | 210,156 |

width:

- min: **9.80**
- median: **30.00**
- mean: **29.75**
- max: **62.99**

median waypoint:

- min: **6**
- median: **7**
- max: **7**

대부분의 stroke는 최대 7 waypoint로 downsample되어 있다. 일부 stroke는 skeleton 구조상 6 waypoint로 남는다.

### Coordinate System

`strokes_medianized.jsonl`의 `median` 좌표는 math-up 좌표계다.

- x: 0..1024 근처
- y: 0..1152 근처
- `y_pivot = 1152`

즉 원본 SVG y-down 좌표를 그대로 둔 것이 아니다. 합성 엔진에서 y-up rasterizer가 바로 읽도록 맞춘 결과다.

KanjiVG와 같은 개념이지만 scale이 다르다.

| Source | viewbox | y_pivot |
|---|---:|---:|
| KanjiVG | 109 x 109 | 109 |
| e-hanja online medianized | 1024 x 1152 | 1152 |

따라서 두 source를 섞어 렌더링할 때는 source별 scale normalization이 필요하다.

## Processing Pipeline

가공 스크립트는 [../db_mining/RE_e-hanja_online/scripts](../db_mining/RE_e-hanja_online/scripts)에 있다.

### Phase 0. Watermark Strip

스크립트:

- [strip_watermark.py](../db_mining/RE_e-hanja_online/scripts/strip_watermark.py)

역할:

- raw SVG를 `db_src/e-hanja_online/svg/`로 복사
- `©2020.(e-hanja)` watermark text node 제거
- 원본 crawl directory는 수정하지 않음

주의:

- SVG geometry 자체를 바꾸는 단계가 아니다.
- text watermark만 제거한다.

### Phase 1. Classification

스크립트:

- [classify_svgs.py](../db_mining/RE_e-hanja_online/scripts/classify_svgs.py)

입력:

- `db_mining/RE_e-hanja_online/data/svg/{HEX}.svg`

출력:

- [strokes_manifest.jsonl](./e-hanja_online/strokes_manifest.jsonl)

판별:

- `class="ani-svg"`가 있으면 `animated`
- 아니면 `static`
- animated일 때 `stroke-normal|stroke-radical` path 수를 `stroke_count`로 기록

### Phase 2. Outline Extraction

스크립트:

- [extract_strokes.py](../db_mining/RE_e-hanja_online/scripts/extract_strokes.py)

입력:

- [strokes_manifest.jsonl](./e-hanja_online/strokes_manifest.jsonl)
- raw SVG

출력:

- [strokes_animated.jsonl](./e-hanja_online/strokes_animated.jsonl)
- [extract_anomalies.jsonl](./e-hanja_online/extract_anomalies.jsonl)

처리:

- manifest에서 `type == "animated"`만 읽음
- outer `<g transform="...">` 추출
- `class="stroke-normal|stroke-radical"` path 추출
- `id="U{CP}d{N}"`에서 stroke order 추출
- `d` path string을 보존

### Phase 5. Medianization

스크립트:

- [medianize_outlines.py](../db_mining/RE_e-hanja_online/scripts/medianize_outlines.py)

입력:

- [strokes_animated.jsonl](./e-hanja_online/strokes_animated.jsonl)

출력:

- [strokes_medianized.jsonl](./e-hanja_online/strokes_medianized.jsonl)
- [medianize_anomalies.jsonl](./e-hanja_online/medianize_anomalies.jsonl)

처리 개념:

1. SVG `d` path를 vertex array로 flatten
2. outer transform 적용
3. y-up 좌표계로 변환
4. stroke outline polygon을 local raster mask로 그림
5. skeletonize로 1-pixel centerline 복원
6. skeleton graph에서 backbone polyline 추적
7. distance transform으로 width 추정
8. 최대 7 waypoint로 downsample

이 단계는 유용하지만 lossy하다. 원본 outline의 모든 곡선 정보를 유지하려면 `strokes_animated.jsonl`을 기준으로 삼아야 한다.

## Anomalies And Data Quality

### `extract_anomalies.jsonl`

현재 26 records가 있고, 모두 `only_one_stroke` 계열이다.

예:

```json
{"cp": 19968, "hex": "4E00", "char": "一", "stroke_count": 1, "issues": ["only_one_stroke"]}
```

이것은 반드시 오류라는 뜻은 아니다. `一`, `丨`, `丶`, Kangxi radical 일부처럼 실제로 1획인 글자가 있다. 이 파일은 "검토가 필요한 케이스"를 모아 둔 것에 가깝다.

### `medianize_anomalies.jsonl`

현재 2 records가 있다.

```json
{"cp": 24432, "char": "彰", "issues": ["order=11_exception=ValueError"]}
{"cp": 34340, "char": "蘤", "issues": ["order=18_exception=ValueError"]}
```

즉 medianization 중 두 stroke가 변환 실패했다. 그래서:

- `strokes_animated.jsonl`: 210,158 strokes
- `strokes_medianized.jsonl`: 210,156 strokes

로 2획 차이가 난다.

실제로 학습 데이터를 만들 때 대부분의 영향은 작지만, 특정 문자 `彰`, `蘤`의 완전한 stroke rendering이 중요하다면 outline 원본을 기준으로 재처리하거나 해당 stroke를 수동 확인해야 한다.

## Comparison With Mobile `e-hanja`

| 항목 | mobile `e-hanja` | `e-hanja_online` |
|---|---|---|
| 로컬 위치 | [e-hanja](./e-hanja) | [e-hanja_online](./e-hanja_online) |
| 원천 | Android app DB | online site/API/SVG |
| 중심 형식 | SQLite/CSV | SVG/JSONL/raw HTML |
| 기본 엔트리 수 | `hSchool` 10,932 | manifest 76,013 |
| 한국어 뜻/훈음 | 매우 강함 | tree/detail에 있음 |
| stroke SVG | app `imgData` PNG 계열 | clean SVG + stroke extraction |
| 희귀 한자 coverage | 제한적 | 훨씬 넓음 |
| 현재 매뉴얼 초점 | 관계형 사전 구조 | SVG 획 데이터 |

mobile e-hanja는 사전 테이블 구조가 좋고, online e-hanja는 coverage와 SVG가 좋다.

둘은 대체재라기보다 서로 보완재다. 예를 들어:

- 한국어 설명과 관계형 table join은 mobile e-hanja
- 더 넓은 glyph/stroke coverage는 e-hanja online
- 같은 글자에서 두 source가 모두 있으면 mobile 사전 정보 + online SVG를 결합

이런 식으로 쓰는 것이 자연스럽다.

## Comparison With Other Stroke Sources

### e-hanja online vs MakeMeAHanzi

| 항목 | e-hanja online | MakeMeAHanzi |
|---|---|---|
| 처리 가능 stroke records | 16,329 chars | 9,574 chars |
| 전체 source coverage | 76,013 SVG | 9,574 JSON lines |
| glyph 성격 | 한국 e-hanja 자형 | 중국계 Hanzi 자형 |
| outline path | animated subset에 있음 | `strokes` field에 있음 |
| median | skeletonized 파생 | 원본 `medians` 제공 |
| decomposition | detail/tree 쪽에 일부 | 강함 |
| license clarity | 약함 | 상대적으로 명확 |

e-hanja online은 coverage 확장에 강하다. MakeMeAHanzi는 compact하고 구조가 정리되어 있으며 decomposition이 강하다.

### e-hanja online vs KanjiVG

| 항목 | e-hanja online | KanjiVG |
|---|---|---|
| glyph 성격 | 한국 쪽 자형 | 일본 Kanji 자형 |
| 처리 가능 records | 16,329 medianized | 6,446 median |
| 원본 geometry | outline polygon | centerline path |
| median 품질 | skeleton 복원 결과 | 원본 centerline |
| stroke kind | normal/radical | `kvg:type` 91종 |
| 좌표계 | 1024 x 1152 | 109 x 109 |

KanjiVG의 median은 더 직접적이고 깨끗하다. e-hanja online은 더 넓고 한국 자형에 가깝지만, median은 skeletonization 결과라 후처리 흔적이 있다.

## Recommended Usage

### 1. Coverage Discovery

어떤 글자가 e-hanja online에 있는지 확인할 때는 manifest를 먼저 본다.

```python
import json

manifest = {}
with open("db_src/e-hanja_online/strokes_manifest.jsonl", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        manifest[rec["cp"]] = rec

cp = ord("鑑")
if cp in manifest:
    print(manifest[cp]["type"])
```

### 2. Stroke-Based Rendering

기존 median 기반 renderer를 쓴다면:

- [strokes_medianized.jsonl](./e-hanja_online/strokes_medianized.jsonl)을 사용
- `viewbox = [1024, 1152]`
- `y_pivot = 1152`
- `median` 좌표는 이미 y-up

원본 outline을 직접 fill rendering하고 싶다면:

- [strokes_animated.jsonl](./e-hanja_online/strokes_animated.jsonl)을 사용
- `transform`을 적용
- `d` path를 SVG path parser로 flatten/fill

### 3. Static SVG Fallback

static 59,684자는 획 분리는 안 되지만, 글자 전체 glyph image로는 쓸 수 있다.

가능한 fallback:

- full glyph mask rendering
- class coverage만 늘리는 OCR pretraining
- stroke-level augmentation 없이 affine/noise/style augmentation만 적용

하지만 stroke order나 per-stroke deformation에는 쓰면 안 된다.

### 4. Dictionary Enrichment

현재 `db_src/e-hanja_online`에는 tree/detail을 정규화한 CSV/SQLite가 아직 없다. 그래도 raw 원본은 mining workspace에 있으므로, 나중에 다음 정보를 뽑아 canonical DB를 보강할 수 있다.

- `getHunum`: 한국어 훈음
- `getJahae`: 뜻 list
- `getSchoolCom`: 변이 관계 10종
- detail HTML: 부수, 획수, 모양자, 자원, 영어, 병음, 교육/검정/인명용 분류

이때 mobile e-hanja와 충돌할 수 있으므로, source column을 분리해서 보존하는 것이 좋다.

## Caveats

### 1. Animated Only Is Not Full Coverage

`strokes_manifest.jsonl`은 76,013자를 담지만, stroke-level로 처리 가능한 animated subset은 16,329자다.

문자 coverage를 말할 때 반드시 둘을 구분해야 한다.

- "e-hanja online SVG가 있다" = 76,013자
- "획별 outline이 있다" = 16,329자
- "median stroke로 변환되어 있다" = 16,329자, 단 2획 변환 실패 있음

### 2. Medianized Data Is Derived

`strokes_medianized.jsonl`은 원본이 아니다. outline을 rasterize하고 skeletonize해서 만든 파생 데이터다.

따라서:

- 자연스러운 stroke jitter에는 유용하다.
- 원본 glyph fidelity는 `strokes_animated.jsonl`이 더 좋다.
- skeleton spur, closed-loop 처리, width 추정 오차가 있을 수 있다.

정확한 자형 렌더링이 중요하면 outline layer를 기준으로 확인해야 한다.

### 3. `radical` Is A Visual Annotation

`kind = radical`은 해당 획이 부수 영역에 속한다는 e-hanja SVG annotation이다. 이것을 곧바로 Kangxi radical ID나 semantic radical relation으로 해석하면 안 된다.

예를 들어 "radical stroke"는 렌더링/강조에는 좋지만, 사전적 부수 join은 별도 DB의 부수 필드와 연결해야 한다.

### 4. Raw HTML Is Not Yet A Clean DB

detail HTML은 정보가 풍부하지만 아직 `db_src`에 정규화 테이블로 들어와 있지 않다. HTML block parser를 만들 때는 다음 위험이 있다.

- 페이지 템플릿 변경
- 빈 필드와 누락 필드 구분
- `<span>`, image thumbnail, 링크 구조 섞임
- 같은 정보가 tree JSON과 detail HTML에 중복

따라서 canonical DB에 넣기 전에는 parser와 validation report가 필요하다.

### 5. Legal/Distribution Risk

온라인 수집 자료이므로 외부 배포는 조심해야 한다. 특히 clean SVG도 watermark text만 제거했을 뿐, glyph geometry 자체는 e-hanja 원천 자료다.

내부 학습, 분석, source comparison 용도로 제한하는 것이 현재 가장 안전한 해석이다.

## Practical Lookup Recipes

### 문자에서 clean SVG 찾기

```python
char = "一"
path = f"db_src/e-hanja_online/svg/{ord(char):X}.svg"
```

e-hanja online은 uppercase hex filename을 쓴다.

### 문자에서 medianized stroke 찾기

```python
import json

by_cp = {}
with open("db_src/e-hanja_online/strokes_medianized.jsonl", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        by_cp[rec["cp"]] = rec

rec = by_cp.get(ord("一"))
```

### 문자에서 raw tree JSON 찾기

```python
char = "一"
path = f"db_mining/RE_e-hanja_online/data/tree/{ord(char):X}.json"
```

### Source Priority Example

stroke rendering source를 고를 때는 다음 순서가 실용적이다.

1. e-hanja online medianized
   - 한국식 자형 coverage가 넓음
2. MakeMeAHanzi
   - median 원본이 있고 decomposition도 좋음
3. KanjiVG
   - 일본식 style diversity와 clean centerline
4. e-hanja online static fallback
   - stroke-level이 필요 없을 때만

단, 모델 목표가 일본어 OCR이면 KanjiVG 우선순위가 올라가고, 중국어 OCR이면 MakeMeAHanzi 우선순위가 올라간다.

## Summary

e-hanja online은 **온라인 e-hanja에서 수집한 고coverage SVG/stroke source**다.

핵심만 정리하면:

- clean SVG: **76,013자**
- animated stroke-separable SVG: **16,329자**
- static monolithic SVG: **59,684자**
- outline strokes: **210,158획**
- medianized strokes: **210,156획**
- 좌표계: animated/medianized는 1024 x 1152
- 강점: 한국 쪽 glyph, 넓은 coverage, radical/normal 획 annotation
- 약점: 라이선스 불명확, static은 획 분리 불가, median은 skeletonization 파생

따라서 Sinograph Explorer에서는 e-hanja online을 일반 사전 DB라기보다 **한국 자형 중심의 대규모 stroke/glyph source**로 쓰는 것이 가장 적합하다. mobile e-hanja의 사전 정보와 결합하면, 한국어 의미 정보와 OCR용 glyph coverage를 동시에 보강할 수 있다.
