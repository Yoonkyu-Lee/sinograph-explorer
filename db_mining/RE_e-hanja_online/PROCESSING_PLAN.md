# SVG 가공 계획 — 수집 이후 파이프라인

수집(crawl) 완료 상태의 원본 SVG 76,013건을 악필 생성기에서 쓸 수 있는
형태로 가공하는 **4-phase 파이프라인**.

- **입력**: `db_mining/RE_e-hanja_online/data/svg/{HEX}.svg` (76,013건)
- **출력**: `db_src/e-hanja_online/` (가공 데이터 새 DB)
- **원칙**:
  - `db_mining/`은 코드, `db_src/`는 데이터
  - 기존 `db_src/e-hanja/` (모바일 앱) 와 **독립** — 비교/참고용일 뿐 연동 없음
  - **stroke 분리가 가능한 SVG만 가공**. 단일 path 덩어리는 일단 manifest에만 기록하고 보류

## 관련 문서

- [SITE_ANALYSIS.md](./SITE_ANALYSIS.md) — SVG가 어떤 구조인지 (animated vs static)
- [COLLECTION_PLAN.md](./COLLECTION_PLAN.md) — 수집 완료된 3축 정의
- [../../doc/08_STROKE_SOURCE_COVERAGE_PLAN.md](../../doc/08_STROKE_SOURCE_COVERAGE_PLAN.md) — 상위 motivation (9,574자 → ~28k자 확장)
- [../../synth_engine_v2/ENGINE_V2_DESIGN.md](../../synth_engine_v2/ENGINE_V2_DESIGN.md) — 엔진 아키텍처, `svg_stroke.py`의 위치
- [../../synth_engine_v2/scripts/svg_stroke.py](../../synth_engine_v2/scripts/svg_stroke.py) — MakeMeAHanzi 기반 기존 구현 (median 중심선 방식)

## 두 종류의 SVG — 왜 animated 만 처리하는가

`SITE_ANALYSIS.md` 정찰 결과 e-hanja가 내보내는 SVG는 두 종류.

| 구분 | 판별자 | 구조 | 처리 가능? |
|------|-------|------|---------|
| **animated** | `class="ani-svg"` + `<path class="stroke-normal\|stroke-radical">` per 획 | 획별 outline (닫힌 폴리곤), `id="U{CP}d{N}"`에 획 순서 | ✅ 가공 |
| **static** | `class="svg"`, 단일 `<path class="path-normal">` | 글자 전체 한 path — 획 경계 없음 | ❌ 보류 |

animated는 획 단위 geometry를 우리가 쓸 수 있는 형태로 제공하지만, static은
획이 하나의 path로 녹아있어 **어느 픽셀이 몇 번째 획인지 복원 불가**. 스켈레
톤화로 추정은 가능하나 lossy하고 어느정도 신뢰도가 있는 획 단위 variation을
약속할 수 없음 → **일단 보류**. 나중에 "글자 전체를 단일 outline으로 놓고
morphological variation만 주는 fallback 렌더러"로 확장 가능.

**핵심 설계 결정 — outline을 median으로 변환하지 않는다**

MakeMeAHanzi는 획 중심선(median polyline) + 폭(width)이지만, e-hanja는 outline
(닫힌 폴리곤). 둘을 median 쪽으로 통일하려면 e-hanja outline을 skeletonize 해
야 하는데, skeletonization은 (a) spur가 잘 생기고 (b) 폭 정보가 날아가며 (c)
40k개 × N 획 규모의 오프라인 비용이 큼. 반면 **outline은 이미 렌더가능한 폴
리곤** 이므로 그대로 두고 `PIL.Draw.polygon(fill=...)`로 바로 렌더하면 정보
손실 없음. 따라서:

- e-hanja outline → **그대로 보존**
- 엔진 쪽에서는 MakeMeAHanzi와 **별개 백엔드** (`EHanjaStrokeSource`)로 사용

## 4-phase 파이프라인

### Phase 1 — Classification 스캔

**목적**: 76,013건 중 animated가 실제로 몇 개인지 실측 + manifest 생성.

**입력**: `data/svg/*.svg`
**스크립트**: `scripts/classify_svgs.py` ✅ 작성됨
**출력**: `db_src/e-hanja_online/strokes_manifest.jsonl`

각 레코드 1줄:

```jsonl
{"cp": 38481, "hex": "9451", "char": "鑑",
 "type": "animated", "stroke_count": 22, "viewbox": [1024, 1152]}
{"cp": 177923, "hex": "2B503", "char": "𫔃",
 "type": "static", "stroke_count": null, "viewbox": [1024, 1152]}
```

**판별 로직**: regex로 `class="ani-svg"` 검사. XML 파서 불필요 (포맷 일정).
`stroke-normal` / `stroke-radical` 카운트로 stroke 수 얻음.

**Phase 1 완료 후 실측 보고 항목**:
- animated 개수 (예상: BMP ≈ 28k)
- static 개수 (예상: SMP ≈ 47k)
- stroke_count 통계 (min/median/max/mean) → Phase 2 스코프 판단

**Phase 1 실측 결과** (2026-04-19):

| 항목 | 값 | 비고 |
|---|---:|---|
| animated | **16,329** | BMP 28,382 중 12,053이 static — "BMP=animated" 가정 틀림 |
| static | **59,684** | 예상(47k)보다 12k 많음 |
| stroke min | **1** | 의심 — Phase 2에서 검증 (1획 글자가 실제인지, 파싱 누락인지) |
| stroke median | 13 | 정상 |
| stroke mean | 12.9 | 정상 |
| stroke max | 39 | 鬱(29) 등 획수 많은 글자 감안 타당 |

→ Phase 2 대상 확정: **16,329자**. Phase 2에서 stroke_count=1 케이스 검증 병행.

### Phase 2 — 획별 outline 추출 (animated만)

**목적**: animated SVG를 파싱해 획마다 outline polygon을 뽑아 compact한
형태로 저장.

**입력**: Phase 1 manifest에서 `type == "animated"`인 항목 + 해당 SVG 파일
**스크립트**: `scripts/extract_strokes.py` (TODO)
**출력**: `db_src/e-hanja_online/strokes_animated.jsonl`

레코드 형식 (안):

```jsonl
{"cp": 38481, "hex": "9451", "char": "鑑",
 "viewbox": [1024, 1152],
 "transform": "scale(1,-1) translate(0, -879)",
 "strokes": [
   {"order": 7,  "kind": "normal",   "outline_d": "M602 738L447 738..."},
   {"order": 8,  "kind": "normal",   "outline_d": "M553 608L553 696..."},
   ...
   {"order": 22, "kind": "radical",  "outline_d": "..."}
 ]}
```

**파싱 포인트**:
- `<path id="U9451d7" class="stroke-normal" d="...">` 에서
  - `d{N}` → `order`
  - class → `kind` (normal / radical)
  - `d` 속성값 → `outline_d` (raw SVG path 문자열 그대로)
- 글자 전체의 `<g transform="...">` 도 보존 (좌표계 정렬에 필요)
- viewBox도 레코드마다 (거의 다 `[1024, 1152]` 예상이지만 확실하게)

**저장 형식의 선택**:
- raw SVG `d` 문자열 그대로 → **렌더할 때 svgpathtools 등으로 파싱**
- 장점: lossless, 나중에 Bezier curve까지 쓸 수 있음
- 단점: 렌더 경로에 SVG path parser 의존 발생
- 대안으로 vertex-sampled polyline도 함께 저장 가능 (후순위 — 필요성 생기면 추가)

### Phase 3 — `EHanjaStrokeSource` 백엔드 구현

Phase 2 완료 후 실측 데이터 기반으로 MakeMeAHanzi 와의 관계를 먼저 정리하고,
통합 전략을 확정한 뒤 구현한다.

#### 3.1 두 소스 데이터 비교

| 항목 | MakeMeAHanzi | e-hanja animated |
|------|-------------|------------------|
| 커버리지 | 9,574자 | 16,329자 |
| 획별 outline SVG path | ✅ (`strokes[]`) | ✅ (`strokes[].d`) |
| **획별 medians (중심선)** | ✅ (`medians[]`) | ❌ (없음) |
| 획 kind (normal/radical) | ❌ | ✅ |
| 획 순서 | 배열 index | `id="U{CP}d{N}"` 에서 N 명시 |
| 좌표계 | 0..1024 box, y-flipped (pivot 900) | 1024×1152 viewBox + 글자별 `<g transform>` |
| 라이선스 | MIT (Arphic PL) | e-hanja (©2020, 학술 fair use 내 사용) |

**중요 재발견**: MakeMeAHanzi도 **outline path 를 가지고 있다**. 현재
`svg_stroke.py`는 `medians`만 쓰지만, `strokes` 필드 자체는 e-hanja 와
동질 데이터. 둘 다 "획별 outline SVG path 리스트"로 정규화 가능.

**차이점 요약**:
- MMH는 **medians 까지 제공**해서 "두꺼운 polyline" 렌더가 간단
- e-hanja는 **outline 만 제공** → median 없음 → outline 기반 렌더 필수
- 두 소스의 **유일 공통 분모는 outline path**

#### 3.2 커버리지 overlap 실측 (2026-04-19)

```
MakeMeAHanzi:     9,574자
e-hanja animated: 16,329자
intersection:     7,671자  (MMH 80.1%, e-hanja 47.0%)
MMH only:         1,903자
e-hanja only:     8,658자
union:           18,232자  ← 통합 후 기대 커버리지 (MMH 대비 1.9× 확장)
```

**MMH only 1,903자** 블록별:
- CJK Unified (4E00-9FFF): **1,889** — 일반 중국어 상용 중 e-hanja 미수록
- CJK Ext A: 14

MMH 만 있는 영역은 거의 전부 BMP CJK Unified. e-hanja 는 한국 교육용 + 이체
위주라 중국 간체 전용 / 희소 중국어 글자가 누락.

**e-hanja only 8,658자** 블록별:
- CJK Unified (4E00-9FFF): **6,024** — MMH 미수록 상용 한자 (한국 고유 포함)
- CJK Ext A (3400-4DBF): **1,001**
- CJK Compat (F900-FAFF): **327** — 호환 한자 전체
- Radicals Supp + Kangxi (2E80-2FDF): **320** — 부수/원형 획
- **Ext B SMP (20000-2A6DF): 907** — proposal 의 "희귀 한자 tail" 타겟
- 기타: 79

**e-hanja only 영역이 proposal 목표와 정확히 일치** — Ext A / Compat /
Radicals / Ext B. 이 8,658자는 MakeMeAHanzi 만으로는 절대 못 얻는다.

#### 3.3 통합 전략 결정

**옵션 A — 통합 outline 백엔드 (권장)**
- 단일 `OutlineStrokeSource` 클래스가 MMH `strokes` 또는 e-hanja `strokes` 양쪽 읽음
- 렌더는 SVG path → flatten polygon → `PIL.ImageDraw.polygon(fill=255)` 하나로 통일
- 기존 `SvgStrokeSource` (median 기반)는 **legacy 로 보존** — 제거하지 않음
- 장점:
  - 한 쌍의 ops 구현으로 양쪽 소스 모두 커버
  - 같은 글자(intersection 7,671자)에서 소스 간 일관된 렌더 스타일
  - union 18,232자 전부 단일 파이프라인
- 단점:
  - SVG path parser 필요 (svgpathtools 또는 mini-parser)
  - width_jitter 가 median 방식보다 복잡 (morphological op 필요)

**옵션 B — 두 병렬 백엔드**
- `SvgStrokeSource` (MMH medians) 그대로 유지
- `EHanjaStrokeSource` (outline) 별도 신설
- config 에서 `kind: svg_stroke` 또는 `kind: ehanja_stroke` 선택
- 장점: 각 백엔드를 해당 데이터 특성에 맞춰 최적화
- 단점: 동일 글자 렌더 스타일이 두 소스에서 서로 달라질 수 있음 → 학습 데이터 혼선
  가능. 코드 경로 이중화

**결정: 옵션 A** — 단, **점진적 마이그레이션**.
1. 먼저 e-hanja 전용 outline 백엔드를 추가해서 동작 검증 (Phase 3a)
2. 검증 후 MMH 도 같은 outline 백엔드로 처리하도록 확장 (Phase 3b)
3. 최종적으로 median 기반 `SvgStrokeSource` 는 deprecated 표시만, 제거는 하지 않음
   (참고용·비교용)

이 순서면 Phase 3a 에서 "e-hanja 단독 사용 가능" 이미 성립 → 엔진 기능이 바로
확장됨. Phase 3b 는 refactor 이므로 보상 없이 지연 가능.

#### 3.4 Phase 3a — e-hanja outline 백엔드 (즉시 착수)

**파일**: `synth_engine_v2/scripts/ehanja_stroke.py` (신규)
**인터페이스**: 기존 소스와 동일 — `render_mask(char, rng) -> Image.Image`

**내부 표현**: `OutlineStrokeData`
```python
@dataclass
class OutlineStrokeData:
    char: str
    viewbox: tuple[int, int]    # (w, h), 일반적으로 (1024, 1152)
    transform: str | None       # SVG 원문 transform (렌더 시 행렬로 변환)
    strokes: list[OutlineStroke]

@dataclass
class OutlineStroke:
    order: int                  # 획 순서 (1-indexed)
    kind: str                   # "normal" 또는 "radical"
    polygon: np.ndarray         # shape (N, 2) — flatten 된 vertex 좌표
```

**로딩**:
- 생성 시점에 `strokes_animated.jsonl` 한 번 읽어서 char→record dict 로 메모리
  상주 (약 16k 레코드, ~50MB 이내 예상)
- 각 stroke 의 SVG path `d` 를 **flatten 한번만** (lazy cache 가능)
- flatten 은 Bezier 곡선을 small segment 로 샘플링 → polygon vertex array

**SVG path parsing 선택**:
- 1순위: `svgpathtools` 사용 (pip install svgpathtools, 대략 50KB 모듈)
  - 장점: 검증된 파서, Bezier 지원
- 2순위: mini-parser (M/L/Q/C/Z 만 지원) — 대부분의 stroke 가 이 범위 내
  - 장점: 외부 의존성 0
  - 단점: 예외 문자 마주치면 실패

Phase 3a 에서는 **svgpathtools 권장** (구현 속도 우선).

**좌표 변환**:
- e-hanja: viewBox (0, 0, 1024, 1152), `<g transform="scale(1,-1) translate(0,-879)">`
- 파싱한 vertex 에 transform 적용 후, `(1024, 1152)` 박스 기준 정규화
- 엔진 캔버스 (`CANVAS=384`, `PAD=48`) 로 스케일 + 중앙 정렬
- **정렬 결과**: `MMH_BOX=1024` 와 동일한 최종 캔버스 위치로 떨어뜨림 → 나중에
  MMH 출력과 나란히 비교 가능

**Per-stroke ops (outline 버전)**:

| op | outline 처리 |
|----|------------|
| `stroke_rotate` | 획 centroid 기준 회전 행렬 적용 |
| `stroke_translate` | 모든 vertex 에 (dx, dy) 추가 |
| `control_jitter` | 중간 vertex 에 Gaussian noise |
| `drop_stroke` | 해당 폴리곤 skip |
| `width_jitter` | 획 별도 렌더 후 morphological dilate/erode (radius Gaussian) |
| `endpoint_jitter` | 휴리스틱 — 획 bbox 장축 방향 양극단 N% vertex 에만 noise |

(endpoint_jitter 는 median 버전보다 자연스러움 열세. Phase 4 에서 실측 후 파라
미터 튜닝)

**렌더**:
```python
img = Image.new("L", (CANVAS, CANVAS), 0)
draw = ImageDraw.Draw(img)
for stroke in varied.strokes:
    pts = stroke.polygon  # after stroke ops
    draw.polygon([tuple(p) for p in pts], fill=255)
# (width_jitter 있으면 획 마스크를 morphological op 적용한 뒤 union)
```

#### 3.5 Phase 3b — MMH 도 outline 경로로 (후속)

- `OutlineStrokeSource` 에 MMH 어댑터 추가: `graphics.txt` 의 `strokes[]` 를 같은
  `OutlineStrokeData` 로 로드
- 좌표계: MMH 는 (0..1024 box, y-flip pivot=900) → 정규화 함수로 통일
- MMH 용 stroke kind 는 모두 `"normal"` (radical 구분 없음)
- 기존 `SvgStrokeSource` 는 **그대로 두고 사용도 계속 가능** — deprecated 주석만 추가

#### 3.6 출력 위치 요약

```
synth_engine_v2/scripts/
  svg_stroke.py          # 기존 median 기반 (legacy, 유지)
  ehanja_stroke.py       # Phase 3a — outline 기반 (신규)
  # Phase 3b 완료 시:
  #   outline_stroke.py  # 통합 outline 백엔드
  #   ehanja_stroke.py   # thin adapter → outline_stroke
  #   svg_stroke.py      # deprecated 주석, 그대로 동작
```

### Phase 4 — 엔진 통합 + 커버리지 측정

**목적**: 엔진에서 실제 사용 가능하게 만들고, 커버리지 증분을 수치로 공개.

- `generate.py`에 `base_source.kind: ehanja_stroke` 분기 추가
- `discover_ehanja_stroke_sources(char, manifest_path, stroke_ops)` factory
- CLI 사용 예:
  ```bash
  python generate.py 鑑 --config configs/ehanja_handwriting.yaml
  ```
- 커버리지 리포트:
  - MakeMeAHanzi 기반: **9,574자**
  - e-hanja animated: **~28k자** (Phase 1 실측 후 확정)
  - 합집합 / 순수 증분 수치 공개

## 산출물 디렉토리 구조

```
db_src/e-hanja_online/
  strokes_manifest.jsonl       # Phase 1 — 전수 분류
  strokes_animated.jsonl       # Phase 2 — animated SVG 전부의 획별 outline
  README.md                    # (TODO) 포맷 설명
```

## 스코프 밖

- **static SVG 47k의 가공**: 이번 범위 아님. 필요해지면 별도 fallback 렌더러로.
- **SVG 원본의 db_src 복제**: 원본은 `db_mining/.../data/svg/` 에 그대로 남김.
  `db_src/e-hanja_online/`에는 **가공 산출물만** 들어감.
- **canonical DB v1 통합**: 이번 범위 아님. 엔진 소스로 쓰는 게 먼저. canonical
  DB 통합은 엔진이 안정된 뒤 별도 작업.

## 현재 상태

- Phase 1 ✅ 완료 (2026-04-19): 16,329 animated / 59,684 static 분류 완료. manifest 저장.
- Phase 2 ✅ 완료 (2026-04-19): animated 16,329자의 획별 outline 추출 완료
  (`strokes_animated.jsonl`). 단일획 글자 26건 검증 — 모두 legit (부수/원형 획).
- Phase 2.5 ✅ 완료 (2026-04-19): 원본 SVG 워터마크 스트립해서
  `db_src/e-hanja_online/svg/` 에 76,013건 보존.
- Phase 3a ✅ 완료 (2026-04-19): e-hanja outline 백엔드 초기 구현. 鑑 렌더 검증.
  (초기엔 `ehanja_stroke.py` 로 구현)
- Phase 3b ✅ 완료 (2026-04-19): MMH outline 어댑터 추가 + `outline_stroke.py` 로 통합.
  `ehanja_stroke.py` 삭제 — 같은 백엔드가 MMH `strokes[]` / e-hanja animated
  JSONL 양쪽을 단일 `OutlineStrokeSource` 로 다룸. 鑑 MMH/e-hanja 렌더 비교 성공.
  기존 `svg_stroke.py` (median 기반) 는 deprecated 주석만 추가, 그대로 유지.
- Phase 4 ✅ 완료 (2026-04-19): `coverage_report.py` 작성 및 집계.
  MMH 9,574 ∪ e-hanja 16,329 = **18,232자** (MMH 대비 1.9× 확장).
  `synth_engine_v2/out/coverage_report.json` + `coverage_per_char.jsonl` 저장.
- Phase 6 ✅ 완료 (2026-04-19): **KanjiVG 통합**. 6,703 일본 kanji + hiragana/katakana.
  `db_src/KanjiVG/` 클론 (CC-BY-SA 3.0), `extract_kanjivg_medians.py` 작성 —
  KanjiVG path 는 이미 centerline 이므로 skeletonization 불필요, path flatten
  만으로 median 추출 (10.9초, 8 workers). 출력 `strokes_kanjivg.jsonl`.
  svg_stroke 에 `load_kanjivg_median_data` (e-hanja median 로더 alias) +
  `discover_kanjivg_median_sources` 추가. generate.py `kind: kanjivg_median`.
  `kanjivg_handwriting.yaml` config (viewbox=109, width_scale=1.7, jitter 값 1/10 스케일).
  KanjiVG-only 750자 (일본 shinjitai 491 + Hiragana 90 + Katakana 94 등).
  union 최종: **18,982자** (MMH 대비 1.98×).

- Phase 5 ✅ 완료 (2026-04-19): e-hanja outline → **median 스켈레톤화**.
  동기: outline 기반 `control_jitter` 가 vertex 밀도 때문에 "그래픽 노이즈" 느낌,
  반면 median 기반 (svg_stroke) 는 획 단위 의미론적 변주 → 자연스러운 필체.
  `medianize_outlines.py` 작성 (scikit-image + multiprocessing 8 workers,
  전체 16,329자 42초). 출력: `strokes_medianized.jsonl` — MMH 호환 포맷.
  svg_stroke 확장: box/y_pivot 필드, `load_ehanja_median_data`,
  `width_scale` / `width_override` 파라미터 (e-hanja 실측 width 가 MMH default
  48 보다 얇아서 1.5× 권장).
  generate.py 에 `kind: ehanja_median` 신설. `ehanja_median_handwriting.yaml`
  샘플 config. 鑑 + 6 한국 고유 한자 렌더 검증 — MMH svg_stroke 와 동등한 자연스러운 필체 스타일로 e-hanja 전용 커버리지 8,658자 에 확장.
