# e-hanja online — 수집 작업 공간

온라인 e-hanja (`www.e-hanja.kr` + `img.e-hanja.kr` + `tool.img.e-hanja.kr`)로부터
글자별 데이터를 수집하는 작업 공간.

## 문서

- [SITE_ANALYSIS.md](./SITE_ANALYSIS.md) — 사이트 구조·엔드포인트·필드 요약
- [COLLECTION_PLAN.md](./COLLECTION_PLAN.md) — 수집 우선순위·전략·Phase 구분
- [PROCESSING_PLAN.md](./PROCESSING_PLAN.md) — 수집 이후 SVG 가공 4-phase 파이프라인
- [../../doc/09_EHANJA_ONLINE_REVERSING.md](../../doc/09_EHANJA_ONLINE_REVERSING.md) — 정찰 로그
- [../../doc/08_STROKE_SOURCE_COVERAGE_PLAN.md](../../doc/08_STROKE_SOURCE_COVERAGE_PLAN.md) — 상위 motivation

## 디렉토리

```
RE_e-hanja_online/
  SITE_ANALYSIS.md
  COLLECTION_PLAN.md
  README.md  (이 파일)
  samples/        # 정찰 중 수집한 참고 샘플
  scripts/
    codepoints.py    # Unicode 블록 enumeration
    crawl.py         # 메인 크롤러 (3축, resumable)
    inspect_svg.py   # SVG 구조 점검 (단발성 분석 스크립트)
    extract_blocks.py # detail HTML 필드 추출 예시
  data/           # (크롤 시 자동 생성)
    svg/
    tree/
    detail/
  logs/           # (크롤 시 자동 생성)
    crawl.log.jsonl
```

## 사용법

### 준비

```bash
cd d:/Library/01\ Admissions/01\ UIUC/4-2\ SP26/ECE\ 479/lab3
.venv/Scripts/python -m pip install httpx        # 1회만
cd db_mining/RE_e-hanja_online
```

### Smoke test (50자로 빠르게 확인)

```bash
../../.venv/Scripts/python scripts/crawl.py --axis svg    --limit 50
../../.venv/Scripts/python scripts/crawl.py --axis tree   --limit 50
../../.venv/Scripts/python scripts/crawl.py --axis detail --limit 50
```

각 명령이 몇 초~1분 내 끝나야 정상. 에러 발생 시 `data/<axis>/<cp>.err` 파일 확인.

### 본격 실행 (3축 순차)

```bash
# 축별로 수시간, 순차 실행하면 ~12~15시간 (백그라운드 추천)
../../.venv/Scripts/python scripts/crawl.py --axis svg
../../.venv/Scripts/python scripts/crawl.py --axis tree
../../.venv/Scripts/python scripts/crawl.py --axis detail
```

### 본격 실행 (3축 병렬 — 더 빠름)

```bash
# 세 축 동시 실행 — 도메인이 달라 간섭 없음, 전체 ~5~8시간 예상
../../.venv/Scripts/python scripts/crawl.py --axis all --parallel
```

### 중단/재개

- 언제든 **Ctrl+C**로 안전 종료 (in-flight 요청만 끝나고 정지)
- 다시 같은 명령 실행하면 **기존 파일 스킵하고 이어서** 진행
- 재개 판정: `{cp:X}.svg` / `.json` / `.html` / `.404` / `.err` 중 어느 하나라도 있으면 스킵

### 크래시·예외 상황에서의 복구 보장

| 상황 | 보장 |
|------|-----|
| Ctrl+C 정상 종료 | 완벽 재개 — in-flight만 끝나고 stop |
| HTTP 요청 중 크래시 (OOM, 네트워크 등) | 그 cp는 marker 없음 → 다음 실행에서 재시도 |
| 전원 차단·강제 종료 | 동일. 진행 중이던 cp부터 재시도 |
| **파일 쓰기 *도중* 크래시** | **atomic write** 적용 — `.tmp` 파일에 먼저 쓴 뒤 rename. 최종 파일은 완전하거나 아예 없음. 부분 파일 존재 불가 |
| `.tmp` 찌꺼기 남음 | 다음 실행 시작 시 자동 청소 |

### `.err` 마커 처리

transient 실패(네트워크 flap, 일시적 500 등)로 `.err` 마커가 붙었다면:

```bash
# 모든 .err 마커를 지우고 해당 cp들만 재시도
../../.venv/Scripts/python scripts/crawl.py --axis svg --retry-errors
../../.venv/Scripts/python scripts/crawl.py --axis all --parallel --retry-errors
```

### 수동으로 특정 cp 재시도하기

특정 cp만 다시 받고 싶으면 해당 marker/파일 삭제 후 재실행:

```bash
# 예: U+9451 (鑑) 의 svg 재수집
rm data/svg/9451.svg data/svg/9451.404 data/svg/9451.err 2>/dev/null
../../.venv/Scripts/python scripts/crawl.py --axis svg
```

### 옵션

```
--axis {svg,tree,detail,all}   수집할 축 (default: all)
--concurrency N                축 내 동시 worker 수 (default: 3)
--delay SEC                    worker당 요청간 delay (default: 0.3)
--parallel                     축들을 병렬로 실행 (default: 순차)
--limit N                      codepoint 개수 제한 (smoke test용)
--out DIR                      출력 루트 (default: 이 폴더)
--retry-errors                 .err 마커를 지우고 해당 cp 재시도
--breaker-window N             circuit breaker 슬라이딩 윈도우 (default 50)
--breaker-threshold M          N개 중 M개 이상 에러 시 중단 (default 30)
```

## 안정성 / 장애 대응

### 자동 방어 로직 3단

1. **요청 타임아웃** (20s): httpx 레벨. 한 요청이 무한정 매달리지 않음.
2. **cp별 3회 재시도** (exponential backoff 0.5 → 1.0 → 2.0s). 실패 시 `.err` 기록하고 다음 cp.
3. **Circuit breaker**: 최근 50개 중 30개가 error면 **축 전체 중단** + 현황 리포트 출력.

### 특수 상황 처리

- **HTTP 429 / 503 (rate limit)**: 단순 error 아니라 **30s → 60s → 90s 장기 backoff**
- **detail 축 세션 만료**: worker가 5회 연속 error 만나면 세션 자동 재수립 시도
- **`.tmp` 찌꺼기**: 이전 쓰기 크래시로 남은 파일 실행 시작 시 자동 청소

### 중단 시 출력 예 (breaker 발동 또는 Ctrl+C)

```
============================================================
FINAL REPORT
============================================================

[svg]  state=TRIPPED
  disk: done=7,234  notfound=45  err=67  pending=68,750  total=76,096
  session: done=7230 notfound=45 error=67
  last success: U+9451  at  2026-04-18 16:32:10
  ⚠️  BREAKER TRIPPED: 35/50 consecutive-window failures
     top error patterns in window:
       [28]  HTTP 429 (rate-limited)
       [ 5]  timeout
       [ 2]  [Errno 11001] getaddrinfo failed

------------------------------------------------------------
재개 방법:
  (A) pending만 이어받기 (.err는 스킵):
      python crawl.py --axis all --parallel
  (B) .err도 재시도 (transient 실패 다시 도전):
      python crawl.py --axis all --parallel --retry-errors

상세 보고서 저장: logs/abort_report.json
```

- **state=OK**: 정상 완료 또는 수동 Ctrl+C
- **state=TRIPPED**: 서버 측 문제 감지로 자동 중단
- **`disk:` 줄은 실제 저장된 파일을 스캔한 authoritative 값** (세션 카운터와 무관)

### Breaker 조정

서버가 기본값보다 민감하면 완화:
```bash
# 덜 예민하게 (50개 중 45개 이상 에러 시에만 중단)
python crawl.py --axis all --parallel --breaker-threshold 45
```

의심스러우면 엄격하게:
```bash
# 더 빠르게 감지 (20개 중 10개만 에러여도 중단)
python crawl.py --axis all --parallel --breaker-window 20 --breaker-threshold 10
```

## 산출물

```
data/svg/{cp:X}.svg           # SVG composite (일부 chars에만 존재)
data/svg/{cp:X}.404           # 해당 cp는 e-hanja 미커버
data/svg/{cp:X}.err           # 재시도 실패 (수동 검토 필요)

data/tree/{cp:X}.json         # { unicode, getHunum, getJahae, getSchoolCom }

data/detail/{cp:X}.html       # jajun_contentA.asp 원본 HTML

logs/crawl.log.jsonl          # 매 요청 단위 {axis, cp, status, bytes, ms, worker, ts}
```

예상 최종 규모:
- svg: 약 71,716 × 평균 8KB = **~600 MB**
- tree: 약 71,716 × 평균 1KB = **~70 MB**
- detail: 약 71,716 × 평균 7KB = **~500 MB**

총 **~1.2 GB**.

## 진행 상황 모니터링

크롤 중 30초마다 표준출력에 진행 상황 찍힘:
```
  [svg] processed=5000/71716  done=4850  notfound=150  error=0  rate=2.15/s  eta=9.2h
```

외부에서 상태 보려면:

```bash
# 완료된 글자 수
ls data/svg/*.svg | wc -l
ls data/tree/*.json | wc -l
ls data/detail/*.html | wc -l

# 최근 로그 tail
tail -20 logs/crawl.log.jsonl

# 에러 통계
grep '"status": "error"' logs/crawl.log.jsonl | wc -l
```

## 주의

- 사이트 부담 줄이기 위해 **rate ≤ 3 req/sec 유지** 권장
- User-Agent에 연락처 + 학술 목적 명시되어 있음 (crawl.py 상단 `USER_AGENT`)
- 수집 데이터는 **프로젝트 내부 학습 전용**, 외부 재배포 금지
- 500/429 등 응답 연속 발생 시 즉시 중단하고 재검토
