# Stroke-source Coverage — Sub-side Quest Plan

**상태**: 조사 진행 중 / 결정 보류
**우선순위**: **높음** (midpoint 블로커는 아니지만 final demo "희귀 한자 OCR" 주장을 뒷받침하는 데이터 측면의 결정적 소스)
**업데이트**: 2026-04-18 — 온라인 e-hanja 실제 커버리지가 **71,716자** (Ext B 42,711 포함)로 확인되어 이 작업의 가치가 크게 상승. 우선순위를 "중간"에서 "높음"으로 조정.

## 퀘스트 트리 위치

```
Main Quest: 희귀·변이 한자 OCR 시스템
  └─ Side Quest: Stage 1 synthesis engine (v2 완성)
      └─ Sub-quest: 획 단위 variation 모듈 구현
          └─ Sub-sub-quest: ★ 획 데이터 소스 커버리지 확대 (이 문서)
```

## 한 줄 요약

**폰트 기반은 canonical DB 전체(103k)를 다룰 수 있지만, 획 단위 variation은
MakeMeAHanzi 기반이라 9,574자(중국 간체/번체 상용 중심)만 커버.
이 편향을 해소할 추가 이미지 기반 소스를 조사·통합한다.**

## 배경

### 현재 상태 (synth_engine_v2)

| 소스 | 커버리지 | 단위 | 용도 |
|-----|---------|-----|-----|
| FontSource (시스템 폰트) | 폰트가 커버하는 전체 (100k+) | 픽셀 마스크 | 일반 glyph 생성 |
| SvgStrokeSource (MakeMeAHanzi) | **9,574자** | 획별 polyline | **악필/필체 variation** |

### 왜 이 격차가 문제인가

프로젝트 proposal의 핵심 타겟은:
- **희귀 한자** (rare CJK Extension B/C/D/E/F/G)
- **한국 고유 한자** (媤, 乶, 畓 같은)
- **이체자 / 변형체**
- **stylized / 게임체**

이 중 "한국 고유 + SMP 확장 희귀" 영역은 **MakeMeAHanzi 커버리지 밖**. 즉 **획 단위 악필 variation이 가장 필요한 글자들에 적용 불가능**. 정작 tail의 tail이 뚫려 있는 구조.

FontSource는 이 영역을 부분적으로 덮지만, **폰트는 결정적 렌더**라 "같은 글자의 다른 손글씨 분포"를 만들 수 없음. 이 종류의 variation을 만들려면 **획 단위 데이터가 필수**.

## 조사 진행 상황

### 확인된 공개 stroke geometry DB

| DB | 커버리지 | 포맷 | 지역 편향 | 상태 |
|----|---------|-----|---------|-----|
| MakeMeAHanzi | 9,574 | SVG path + medians | 중국 간체/번체 상용 | **통합 완료** |
| KanjiVG | ~6,500 | SVG (stroke-type 메타 포함) | 일본 상용·변형 | 미통합 |
| e-hanja (모바일 DB) | 10,932 | PNG 획순 프레임 | 한국 교과 | **부분 보유** (canonical DB 내 metadata만, 이미지는 추출 안 됨) |
| e-hanja (**온라인**) | **71,716** | **GIF + SVG 135×135** | 한국 교과 + 이체 + SMP rare 풍부 | **미조사** |

### 온라인 e-hanja 커버리지 상세 (2026-04-18 사이트 안내문 확인)

> 유니코드 ver 13.0의 일부 한자 (E, F 제외. 76,013자 중 **71,716자**)

| Unicode 블록 | 블록 전체 | e-hanja 커버 | 커버율 |
|------------|---------|------------|-------|
| 통합 한자 (BMP CJK Unified) | 20,989 | 20,950 | 99.8% |
| 호환 한자 | 472 | 472 | 100% |
| 호환 한자 보충 | 542 | 542 | 100% |
| 부수 한자 | 329 | 21 | 6.4% |
| **확장 A** | 6,592 | 6,582 | 99.8% |
| **확장 B** | 42,718 | **42,711** | **99.98%** |
| 확장 C | 4,149 | 376 | 9.1% |
| 확장 D | 222 | 62 | 27.9% |
| (확장 E, F 미수록) | — | 0 | — |

**핵심 함의**:
- CJK 통합 + 확장 A는 거의 완전 커버 (각 99.8%)
- **확장 B 42,711자** — 42k 희귀 SMP 한자에 대해 한국 canonical glyph 이미지 존재
- 우리 프로젝트의 "long-tail 희귀 한자" 타겟이 이 42k에 거의 전부 포함됨
- 확장 C/D 커버리지는 낮지만 제한적으로 있음

### e-hanja 관련 발견

#### (a) 모바일 DB의 한계 확정

- `imgData` 테이블: 10,932 한자 × 획순 PNG (평균 11 frame/글자)
- SVG 없음. PNG만.
- **완성 한자 이미지 = 마지막 획 프레임** (별도 테이블 아님)
- 커버리지가 **hSchool 10,932**로 제한됨 (앱 용량 238MB 제약으로 추정되는 의도적 subset)

#### (b) 온라인 사이트의 풍부함 확인 (핵심 발견)

`鑑` 한 글자의 variant graph 비교:

| 소스 | 포함된 변이 수 |
|-----|-------------|
| 모바일 e-hanja hSchool | 2 (鑑, 鑒) |
| 우리 canonical DB enriched family | 5 |
| **온라인 e-hanja (확인됨)** | **7** (鑑, 鑒, 鉴, 鍳, 鑬, 𨰲[Ext B], 𫔃[Ext E]) |

온라인 사이트는:
- **SMP CJK Extension B/E 희귀 글자까지** 자체 canonical 이미지 제공 (©2020 e-hanja 워터마크 확인)
- 관계 유형이 세분화 (동자/와자/본자/간체자/약자)
- 사용자 제공 스크린샷에 **"SVG 135×135 픽셀"** 명시됨

#### (c) 모바일 ⊊ 온라인 — 구조적 차이

- 모바일은 교육 앱 용량 제약 내에서 필수 데이터만 담음
- 온라인은 서버 DB 용량 제약 훨씬 완화 → SMP rare char 포함 풍부한 셋 가능
- **이 격차가 얼마나 큰지는 아직 측정되지 않음** (남은 최대 조사 항목)

## 핵심 남은 질문

| # | 질문 | 왜 중요한가 | 조사 방법 |
|--|-----|-----------|---------|
| Q1 | 온라인 e-hanja 총 커버리지는? | 10,932자 이하면 가치 낮음, 그 이상이면 핵심 소스 | 사이트 인덱스/통계 페이지 확인 또는 샘플링 |
| Q2 | SVG URL이 예측 가능한 패턴? | 크롤링 난이도 결정 | DevTools Network 탭 |
| Q3 | 백엔드가 JSON API? SPA? | 크롤 vs API 재현 vs headless 결정 | DevTools |
| Q4 | robots.txt / ToS 정책? | 작업 가능 여부 | 직접 조회 |
| Q5 | Rate limit 존재? | 작업 시간 및 전략 결정 | 경험적 측정 |
| Q6 | 최신 모바일 앱 버전에 변화 있는가? | 재리버싱 가치 판단 | info API 호출 |

## 결정 경로 (3가지)

### 경로 A — 온라인 사이트 직접 크롤링
**조건**: 서버 렌더링 HTML + 정적 SVG URL
**작업량**: 20k 글자 가정 시 약 5~7일 polite crawl (1 req/sec)
**얻는 것**: SVG 벡터 획 데이터 + 풍부한 변이 그래프 + rare char 커버리지
**리스크**: robots.txt / ToS 준수 필요

### 경로 B — 웹 API 재현
**조건**: DevTools에서 JSON API 발견
**작업량**: 경로 A의 절반
**얻는 것**: 경로 A와 같지만 깨끗한 JSON 형태
**리스크**: 경로 A보다 낮음

### 경로 C — 최신 모바일 앱 재리버싱
**조건**: 새 버전에 유의미한 coverage/format 변화
**작업량**: 기존 도구 재활용 가능 (jadx, apktool, SQLCipher), 약 1~2일
**얻는 것**: 추정상 여전히 PNG + subset. 근본 해결 안 됨
**리스크**: 낮음 (완전 로컬 작업)
**평가**: **fallback only** — 온라인 경로가 전부 막힐 때만

### 추천 순서

```
Step 1 (5분): DevTools 정찰
  ├─ robots.txt 확인
  ├─ Network 탭: 검색 시 발생 요청 관찰
  └─ Elements 탭: 이미지 자산 형식 (<img src=svg>, inline svg, <object>)

Step 2 (정찰 결과 분기)
  ├─ Static HTML + 정적 SVG URL    → 경로 A
  ├─ JSON API 발견                  → 경로 B
  ├─ JS SPA                         → 경로 C' (headless browser 크롤)
  └─ 전부 봉쇄 / 법적 제약          → 경로 C (모바일 재리버싱)

Step 3 (경로 결정 후): 실제 추출 파이프라인 구축
```

## 성공 기준

- [ ] 온라인 e-hanja 커버리지 측정 (Q1 답)
- [ ] SVG 추출 방법 확정 (Q2~Q5 답)
- [ ] 선택된 경로로 파일럿 수집 (鑑 + 주변 이체자 10개 정도)
- [ ] 수집된 SVG에서 polyline 추출 검증
- [ ] SvgStrokeSource에 추가 backend로 통합
- [ ] 통합 후 통계 공개 (e-hanja 기반 커버리지 증분)

## 의존성 / 연관 문서

- [ENGINE_V2_DESIGN.md](../synth_engine_v2/ENGINE_V2_DESIGN.md) — 엔진 아키텍처, svg_stroke 모듈 위치
- [svg_stroke.py](../synth_engine_v2/scripts/svg_stroke.py) — 현재 MakeMeAHanzi 기반 구현
- [CANONICAL_DB_V1_MANUAL.md](../sinograph_canonical_v1/CANONICAL_DB_V1_MANUAL.md) — DB 구조 (이미지 데이터는 현재 `source_payloads.ehanja_raw`에 `imgData` BLOB 제외 형태로만 들어가 있음)
- [db_mining/RE_e-hanja/RE_STATUS.md](../db_mining/RE_e-hanja/RE_STATUS.md) — 모바일 앱 리버싱 결과
- [02_PROPOSAL_DRAFT.md](./02_PROPOSAL_DRAFT.md) — "희귀 한자·이체자" 타겟 근거
- [07_TWO_STAGE_WORKFLOW.md](./07_TWO_STAGE_WORKFLOW.md) — 전체 파이프라인 맥락

## 결정 보류 이유 / 지금 당장 진행 안 하는 이유

**이 작업은 Lab 3 midpoint 블로커가 아니다.** 현재 MakeMeAHanzi 기반 SvgStrokeSource로 9,574자 커버하고, 나머지는 FontSource로 대응 가능. **midpoint 분류기 초기 학습은 현재 엔진으로 시작 가능**.

하지만 **final demo 및 논문/보고서에서 "희귀 한자 OCR"를 주장하려면**, 특히 **한국 고유 한자 영역이 tail에 있다는 proposal의 전제를 실제로 뒷받침하려면**, e-hanja 온라인 데이터 확보가 결정적일 수 있다. 따라서:

- midpoint 이후 여력 있을 때 재개
- Step 1 (5분 정찰)만 먼저 해서 경로 결정 → 실제 작업은 여력 판단 후

## 다음에 여기 돌아올 때 읽을 것

1. 이 문서의 "핵심 남은 질문" 섹션
2. ENGINE_V2_DESIGN.md의 "v2에서도 아직 안 열리는 것" 섹션
3. 가장 최근 canonical DB build_summary.json — e-hanja 기반 covered char 수

## DevTools 정찰 체크리스트 (사용자가 직접 수행)

에이전트 환경에서 www.e-hanja.kr 접속 불가 (HTTPS 미지원 + 샌드박스 제약).
사용자가 직접 브라우저에서 5분 정찰 수행 필요.

**확보해야 할 4개 정보**:

1. **robots.txt** — `http://www.e-hanja.kr/robots.txt` 직접 방문, 전문 복사
2. **Network 요청 패턴** — F12 → Network → Fetch/XHR 필터 → `鑑` 검색
   - URL 형태 (`.asp?`, `/api/`, `/search/` 등)
   - Method (GET/POST)
   - 대표 요청 2~3개
3. **SVG URL 패턴** — 한자 이미지 우클릭 → 이미지 새 탭에서 열기 → 주소 확인
4. **DOM 구조** — 이미지 우클릭 → 검사 → `<img>`/`<object>`/inline `<svg>` 여부

위 4개 확보되면 경로 A/B/C 중 어느 것으로 갈지 확정 가능.

## 변경 이력

- 2026-04-18: 최초 작성. 모바일 vs 온라인 e-hanja 격차 발견 직후 기록.
- 2026-04-18 (개정): 온라인 e-hanja 실제 커버리지 71,716자 (Ext B 42,711 포함) 확인. 우선순위 중간→높음. 정찰 체크리스트 추가.
