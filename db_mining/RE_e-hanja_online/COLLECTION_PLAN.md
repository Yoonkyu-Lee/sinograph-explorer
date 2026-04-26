# 수집 계획

[SITE_ANALYSIS.md](./SITE_ANALYSIS.md) 기반. **Phase 1 MVP 3축** + Phase 2 확장축 정의.

## Phase 1 — MVP 3축 (반드시 수집)

| # | 축 | 엔드포인트 | 세션 | 볼륨 | 요청수 |
|--|---|----------|-----|-----|------|
| 1 | **SVG 이미지** | `img.e-hanja.kr/hanjaSvg/aniSVG/{folder}/{cp}.svg` | 불필요 | ~600 MB | 71,716 GET |
| 2 | **JSON 3종** (tree) | `tool.img.e-hanja.kr/hanjaSvg/asp/dbHandle.comsTree.asp` | 불필요 | ~70 MB | 71,716 × 3 = 215,148 POST |
| 3 | **상세 HTML** | `www.e-hanja.kr/dic/contents/jajun_contentA.asp` | 필요 | ~500 MB | 71,716 POST |

총 요청수: 약 **360k**.
총 용량: 약 **1.2 GB**.

### 왜 이 3축만 MVP인가

- **축 1 (SVG)**: 한국 canonical 자형 — OCR 학습 핵심. 다른 방법으로 대체 불가.
- **축 2 (JSON tree)**: 이미 변이 관계 + 훈음 + 뜻 확보. 가벼움 (300B/응답).
- **축 3 (Detail HTML)**: 14개 필드 — 모양자/자원/영문/교육/급수/인명/유의 등 **다른 축에 없는 유일 정보**. 약간 무겁지만 (~7KB/응답) DB 재구성에 필수.

다른 축은 Phase 2로 연기 (아래 참고).

### 수집 속도 계획

- **동시성**: 3 worker (도메인별 1개씩, 서로 간섭 없음)
- **레이트**: 각 worker 0.3s delay + jitter → 합산 ~3 req/sec
- **타임아웃**: 요청당 20s, 재시도 3회 (exponential backoff)
- **예상 시간**: 360k req ÷ 3 req/s ÷ 3600 = **~33시간** (하루 반 연속 실행)

실제론 빠른 분기(JSON 300B)는 더 빨리 끝나고 느린 분기(HTML 7KB)가 bottleneck. 세 축을 **병렬 실행**하면 전체 소요는 **가장 느린 축 시간에 수렴** (~12시간 내외 예상).

### 재개 가능성 (resume)

- 각 요청 결과를 축별 폴더에 파일로 저장 (`data/svg/{cp:X}.svg`, `data/tree/{cp:X}.json`, `data/detail/{cp:X}.html`)
- 404 응답은 `.404` marker 파일로 기록
- 오류는 `.err` marker + error log
- 재실행 시 **이미 존재하는 파일은 스킵** (marker 포함)
- Ctrl+C 안전 — 다음 실행에서 이어서

### 출력 구조

```
RE_e-hanja_online/
  data/
    svg/
      9451.svg              # composite SVG
      9451.404              # (없으면 marker)
      ...
    tree/
      9451.json             # { hunum, jahae, schoolcom } 3종 통합
      9451.404
      ...
    detail/
      9451.html             # raw jajun_contentA.asp response
      9451.404
      ...
  logs/
    crawl.log.jsonl         # 매 요청별 이벤트 로그 (축, cp, status, bytes, ms)
```

## Phase 2 — 확장 축 (선택적)

Phase 1 완료 후, 필요에 따라:

### P2a. 단어 사전 (word-level)

- `word_search.asp` 경로 — 각 글자 포함 단어 list
- `pop_word.asp?word_id=N` — 단어 상세
- 볼륨: 85k+ word × ~3KB ≈ 250MB
- 용도: 복합어 context OCR 평가용

### P2b. 한자성어

- `word_gosaList.asp` — 성어 전체
- 볼륨: 수천 개 × ~2KB ≈ 10MB
- 용도: 한국 문화 컨텍스트

### P2c. 교차 참조

- `etc_threeHanjaA.asp` — 한중일 3국 비교 (★ 지역 glyph 차이)
- `etc_sameA.asp` — 유의/상대자 쌍
- `etc_yakjaA.asp` / `etc_popularA.asp` — 약자/속자 전체 인덱스
- `etc_subSameA.asp` — 같이 쓰는 한자
- `etc_jajunLenthA.asp` — 장단음
- `etc_sndSokA.asp` — 음 달라진 한자
- `etc_treeHanja.asp` — 이형동의자
- 볼륨: 각 1~10MB 수준, 합산 ~30MB
- 용도: variant graph 보강, OCR 혼동 분석용 semantic 쌍

### P2d. 교육·법적 분류

- `grade_lawA.asp` (대법원 인명용)
- `grade_11A.asp` (교육용 중·고등)
- `grade_20/21A.asp` (검정 읽기/쓰기)
- 볼륨: 수 MB
- 용도: eval tier 구성 (초급/중급/고급)

### P2e. 한자 팝업 (중복 체크용)

- `pop_jajunA.asp` — 상세의 축약 버전
- 대부분 Phase 1의 detail HTML과 중복되므로 **일반적으로 불필요**
- 단, Phase 1 detail HTML 파싱 실패 시 crosscheck용으로 유용

### P2f. 부수/획수/음 인덱스

- `jajun_busu.asp` + AJAX — 부수 tree (325 부수)
- `jajun_number.asp` — 획수 인덱스
- `jajun_sound.asp` — 음 인덱스
- 중복 정보 (Phase 1 데이터에서 파생 가능하지만 원본 pre-grouped 버전 있으면 검증에 좋음)

## 법적·윤리적 가이드

- robots.txt 부재 → 기술적 금지 없지만 **과용 주의**
- 접속 속도 **절대 3 req/sec 상한**
- User-Agent에 **목적 + 연락처** 명시
- 수집 데이터는 **프로젝트 내부 학습 전용**, 외부 재배포 금지
- 사이트 서비스에 부담 주지 않도록 야간/비활성 시간 분산 권장
- 실패(500, 429 등) 감지 시 즉시 backoff 늘리고, 반복되면 중단

## 실행 전 최종 체크리스트

- [ ] `httpx` 라이브러리 설치 (`pip install httpx`)
- [ ] 출력 디렉토리 확보 (약 1.2 GB 여유 공간 필요)
- [ ] 네트워크 안정성 확인
- [ ] Phase 1 3축 스크립트 smoke test (10자 정도)
- [ ] 본격 실행 시작 (대기 ~12시간)
- [ ] 수집 완료 후 **sanity check**: 샘플 수, 용량, 파일 수 집계

## Phase 1 완료 후 후속 작업

1. 수집 데이터 검증 스크립트 작성
2. Phase 1 데이터를 별도 DB 스키마로 정리 (JSON/Parquet/SQLite 중 택1)
3. `canonical_db` 통합 여부는 별도 결정 (현재는 **후순위**)
4. Phase 2 축 추가 수집은 필요시 선택적 진행
