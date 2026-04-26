# e-hanja Online — 사이트 리버싱 작업 로그

> `www.e-hanja.kr` 온라인 사이트의 데이터 추출 경로 조사.
> [08_STROKE_SOURCE_COVERAGE_PLAN.md](./08_STROKE_SOURCE_COVERAGE_PLAN.md) 하위.
>
> **Phase 1 (정찰) 완료.** 경로 결정 및 파일럿 수집 단계로 진입 가능.

## 결론 한 줄

**SVG URL 패턴이 완전히 예측 가능 — 검색 API 우회해서 codepoint enumeration만으로 71,716개 SVG 전체 수집 가능.**

## 확정된 SVG URL 패턴

```
http://img.e-hanja.kr/hanjaSvg/aniSVG/{(cp & 0xFF00):X}/{cp:X}.svg
```

- **서브도메인**: `img.e-hanja.kr` (별도 이미지 서버, 세션 불필요)
- **폴더링**: 256자 단위. codepoint를 0xFF00으로 마스킹한 값.
- **파일명**: codepoint hex (가변 길이, 4~5 자리). **대문자 사용**.

검증된 URL:

| codepoint | URL |
|-----|-----|
| U+9451 (鑑) | `http://img.e-hanja.kr/hanjaSvg/aniSVG/9400/9451.svg` |
| U+5ABA (媺) | `http://img.e-hanja.kr/hanjaSvg/aniSVG/5A00/5ABA.svg` |
| U+28C32 (𨰲) | `http://img.e-hanja.kr/hanjaSvg/aniSVG/28C00/28C32.svg` |
| U+2B503 (𫔃) | `http://img.e-hanja.kr/hanjaSvg/aniSVG/2B500/2B503.svg` |

## SVG 내용 구조 (두 종류)

동일 URL 패턴이지만 내부 구조가 유니코드 블록에 따라 다름.

### Animated SVG — 획 단위 분해 가능

대상: **BMP 통합 + 호환 + Ext A 대부분** (약 28,000자 추정)

특징:
- 루트 `<svg id="U{cp}ani" class="ani-svg">`
- `<style>`로 `opmGna.svg.ani.min.css` import
- `<script>`로 `gna.lib.svg.ani.min.js` import (브라우저 애니메이션용, 우리에겐 무관)
- 획별 `<path id="U{cp}d{N}" d="..." class="stroke-normal|stroke-radical">` N개
- `<clipPath id="U{cp}c{N}">`는 애니메이션 masking용 (중심선 아님)

획별 분해:
- `U{cp}d1, U{cp}d2, ..., U{cp}dN` = 획 순서대로 outline d-path
- `stroke-radical` 클래스로 **부수 소속 획을 라벨링** (예: 鑑 22획 중 처음 8획이 金부수 → stroke-radical)
- **중심선(centerline) 데이터는 없음** — 이 점은 MakeMeAHanzi와 차이

### Static SVG — 획 분리 없음

대상: **Ext B / C / D / E** (약 43,000자)

특징:
- 루트 `<svg id="U{cp}" class="svg">` (ani 접미사 없음)
- 단일 `<path>` element (monolithic shape)
- 획별 ID/class 없음
- 애니메이션 관련 스타일/스크립트 없음

실질적으로 **"한국 정통 자형으로 그려진 완성 이미지"** — 획 조작은 불가능하지만 폰트 렌더 대체로 사용 가능.

### 파일 크기 감각 (샘플 기준)

| cp | 블록 | paths | 크기 |
|----|----|-----|-----|
| U+9451 (鑑) | BMP 통합 | 44 | 10.0 KB |
| U+5ABA (媺) | BMP 통합 | 24 | 6.5 KB |
| U+28C32 (𨰲) | Ext B | 1 | 2.8 KB |
| U+2B503 (𫔃) | Ext E | 1 | 1.8 KB |

전체 71,716자 × 평균 8KB ≈ **약 600 MB** 예상.

## 보조 SVG (누적 프레임)

각 글자마다 **누적 획 진행 프레임** 파일들도 존재:

```
http://img.e-hanja.kr/hanjaSvg/aniSVG/9400/9451-01.svg   ← 획 1만
http://img.e-hanja.kr/hanjaSvg/aniSVG/9400/9451-02.svg   ← 획 1~2
...
http://img.e-hanja.kr/hanjaSvg/aniSVG/9400/9451-22.svg   ← 획 1~22 (=완성)
```

검증: `9451-NN.svg` 파일이 1..N 획을 모두 포함하는 누적 형태 (모바일 DB PNG의 SVG 판).
**수집 불필요** — composite `9451.svg`에 모든 획이 이미 분리 저장돼 있어 composite만으로 충분.

## 기술 스택 (참고)

- Classic ASP + jQuery 3.7.1 AJAX
- 페이지 charset meta: UTF-8
- URL 인코딩: **UTF-8** (`hanja=%E9%91%91` = UTF-8 `鑑`)
- Session: ASPSESSIONID 쿠키 (이미지 서버에선 불필요)
- 우클릭/드래그/복사 차단 (JS-level, HTTP 요청엔 영향 없음)

## API Endpoint Map

### 주 도메인 — `www.e-hanja.kr` (세션 필요, UI-oriented)

| Endpoint | 역할 | 비고 |
|---------|-----|------|
| `/dic/dictionary.asp` | 메인 frameset 내용 | AJAX 기반 UI |
| `/dic/jajun/jajun_search.asp?keyword=...` | 검색 결과 페이지 (shell) | UTF-8 URL 인코딩 |
| `/dic/jajun/jajun_searchA.asp` | 검색 결과 AJAX | POST, session 민감 |
| `/dic/contents/jajun_content.asp?hanja=X` | 상세 페이지 (shell) | UTF-8 |
| `/dic/contents/jajun_contentA.asp` | 상세 AJAX | POST |

### 이미지 서버 — `img.e-hanja.kr` (세션 불필요, 정적 서빙)

| URL 패턴 | 반환 |
|---------|-----|
| `/hanjaSvg/aniSVG/{(cp&0xFF00):X}/{cp:X}.svg` | **메인 composite SVG** (우리가 사용) |
| `/hanjaSvg/aniSVG/{folder}/{cp}-NN.svg` | 누적 획 프레임 (composite에서 유도 가능, 중복) |

### **JSON API — `tool.img.e-hanja.kr`** (핵심 발견)

```
POST http://tool.img.e-hanja.kr/hanjaSvg/asp/dbHandle.comsTree.asp
Content-Type: application/json
Body: {"request": "<type>", "unicode": "U+<hex>", "db": "server"}
```

**세션 불필요.** 단순 POST + JSON body. 모든 블록(Ext B/E 포함)에서 동작.

지원 request type (이체자 관계도 JS에서 역추적):

| request | 반환 내용 | 예시 (U+9451 기준) |
|--------|----------|-----------------|
| `getSchoolCom` | **hSchoolCom 그대로** — 변이 관계 dict | `{dongja: "鍳,鑒,鑬,𨰲", tongja: "", waja: "", bonja: "", ...}` |
| `getHunum` | 훈음 (읽기 + 훈) | `{hanja: "鑑", hRead: "거울 감"}` |
| `getJahae` | 뜻 리스트 (자해) | `[{meaning: "거울", root_snd: "감"}, {meaning: "본보기"}, ... 9개]` |

---

## 온라인 ⟷ 모바일 DB 차이 (실측)

API로 직접 확인한 결과. 이 부분이 오늘 정찰의 **최대 성과**.

### getSchoolCom (변이 관계) — 온라인 레코드

| cp | 글자 | 온라인 응답 | 모바일 보유 |
|-----|-----|----------|-----------|
| U+9451 | 鑑 | `dongja: "鍳,鑒,鑬,𨰲"` (4개) | `dongja: "鑒"` (1개) |
| U+9452 | 鑒 | `waja: "鉴", simple: "鉴", dongja: "鑑"` | 보유 but diff |
| U+9274 | 鉉 | `bonja: "鑒"` | 보유 |
| U+9373 | 鍳 | `dongja: "鑑"` | ❌ hSchool 밖 |
| U+946C | 鑬 | `dongja: "鑑"` | ❌ hSchool 밖 |
| **U+28C32** | 𨰲 (Ext B) | `simple: "𫔃", dongja: "𫔃,鑑"` | ❌ **완전 부재** |
| **U+2B503** | 𫔃 (Ext E) | `dongja: "𨰲"` | ❌ **완전 부재** |

결론:
- **BMP 상용** 글자도 온라인이 평균 3~4배 많은 variant 엣지 보유
- **SMP Ext B/E** 글자에도 관계 데이터 존재 — 모바일엔 아예 없음
- 관계 유형이 **세분화** (mobile은 dongja 주로, 온라인은 bonja/waja/goja/simple/kanji/tongja/yakja/sokja/hDup 까지)

### 이 차이가 프로젝트에 주는 가치

- **canonical DB의 `enriched_family_members`가 대폭 확장** 가능
- 특히 희귀 SMP 글자의 Korean-context variant 관계 — 어디서도 구할 수 없는 데이터
- 모델 학습 후 평가 시 "같은 family 내 오인식"을 부분점수로 처리하는 family-aware metric에 직접 활용

---

## 수집 전략 (Phase 2)

**수집 대상이 3축으로 확장됨**:

1. **SVG (이미지)** — 71,716자 × 평균 8KB ≈ 600 MB (`img.e-hanja.kr`)
2. **getSchoolCom (변이 그래프)** — 71,716자 × 평균 0.3KB ≈ 20 MB (`tool.img.e-hanja.kr`)
3. **getHunum + getJahae (훈음/뜻)** — 71,716자 × 합계 1KB ≈ 70 MB (`tool.img.e-hanja.kr`)

각 축은 **완전 독립 엔드포인트**라 서로 병렬 수집 가능.

### Step 1 — codepoint enumeration 작성

e-hanja가 커버한다고 공개한 유니코드 블록에서 codepoint 리스트 생성:

```python
ranges = [
    (0x3400, 0x4DBF + 1),     # Ext A
    (0x4E00, 0x9FFF + 1),     # BMP 통합
    (0xF900, 0xFAFF + 1),     # 호환
    (0x2E80, 0x2EFF + 1),     # 부수 보충
    (0x2F00, 0x2FDF + 1),     # 강희 부수
    (0x20000, 0x2A6DF + 1),   # Ext B
    (0x2A700, 0x2B73F + 1),   # Ext C
    (0x2B740, 0x2B81F + 1),   # Ext D
    (0x2F800, 0x2FA1F + 1),   # 호환 보충
]
```

총 ~76,000 codepoint → 404 Not Found 제외하고 71,716자 수집 예상.

### Step 2 — polite crawl

```python
async def fetch(cp, sem, session):
    url = f"http://img.e-hanja.kr/hanjaSvg/aniSVG/{cp & 0xFF00:X}/{cp:X}.svg"
    async with sem:
        r = await session.get(url, timeout=15)
        if r.status == 200:
            save(cp, r.text)
        await asyncio.sleep(0.2)  # 1 / 5 req/sec per worker
```

- 동시성: 3 parallel worker
- 대략 5 req/sec 합계 → 71,716 ÷ 5 ÷ 3600 ≈ **4시간**
- 더 보수적으로 가면 1 req/sec 단일 세션 → 약 20시간

### Step 3 — 검증 및 분류

- 다운로드한 SVG를 animated / static으로 분류 (루트 svg class 확인)
- animated에선 획별 d-path 추출 → `db_src/EHANJA_ONLINE_SVG/animated/{cp}.json`
- static은 원본 SVG 그대로 저장
- 추출 통계 산출 (animated 개수, static 개수, 블록별 분포)

### Step 4 — canonical DB 통합

- canonical DB의 각 record에 `media.ehanja_stroke_outline` 필드 추가
- animated만 MMH와 병렬 가능 (per-stroke 조작)
- static은 base_source의 raster fallback으로만 유용

### Step 5 — SvgStrokeSource 확장

- 기존 SvgStrokeSource는 MMH `stroke_medians` 기반 polyline 렌더
- e-hanja outline 지원 추가: outline fill 기반 렌더 + per-stroke variation(translate/rotate/drop)
- width/endpoint jitter는 **centerline이 없어 제한적** — polyline-first 접근이 더 자연스러운 조작 가능

## 법적 / 윤리적 메모

- robots.txt 없음 (명시적 금지 부재)
- 학술 연구 목적으로 한정
- User-Agent에 목적 명시 권장: `"ECE479-Lab3-Research-Crawler/1.0 (+https://...)"`
- 수집 데이터는 **프로젝트 내부 학습 전용**, 외부 재배포 금지
- 이미지 서버(`img.e-hanja.kr`)는 세션 없이 응답 — 과도한 속도 주의

## 작업 로그

### 2026-04-18

**정찰 단계 전부 완료** (curl 기반).

**주요 발견 순서**:
1. curl로 HTTP 접속 가능 확인 (WebFetch는 HTTPS 자동 업그레이드로 차단됨).
2. 프론트엔드가 classic ASP + jQuery, AJAX 기반.
3. 키워드 URL 인코딩은 **UTF-8** 사용 (서버 디코딩은 페이지 생성 단계에서 처리).
4. 검색 API (`jajun_searchA.asp`)는 session 필요 + 인코딩 민감해서 curl만으로 완벽한 호출 어려움.
5. **검색을 우회**하는 결정적 돌파구: 상세 페이지 URL(`jajun_content.asp?hanja=%E9%91%91`) 직접 호출 → AJAX 응답 HTML에서 `img.e-hanja.kr` SVG URL 패턴 추출.
6. SVG URL 구조 확정: `/hanjaSvg/aniSVG/{cp&0xFF00:X}/{cp:X}.svg`. 검색/세션 불필요. 단순 GET.
7. SMP Ext B/E 희귀 글자도 HTTP 200. 단 animated 아닌 monolithic SVG (획 분해 없음).

**다음**: Phase 2 파일럿 수집 착수 (70~100자 배치로 URL 패턴/서버 응답 안정성 재검증).

## 관련 파일

- [../doc/08_STROKE_SOURCE_COVERAGE_PLAN.md](./08_STROKE_SOURCE_COVERAGE_PLAN.md) — 상위 plan
- [../db_mining/RE_e-hanja/svg_samples/](../db_mining/RE_e-hanja/svg_samples/) — 정찰 중 다운받은 샘플 SVG
- [../db_mining/RE_e-hanja/RE_STATUS.md](../db_mining/RE_e-hanja/RE_STATUS.md) — 모바일 앱 리버싱 결과
- [../synth_engine_v2/scripts/svg_stroke.py](../synth_engine_v2/scripts/svg_stroke.py) — 통합 타겟 모듈
