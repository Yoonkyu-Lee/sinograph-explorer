# e-hanja Online — 사이트 분석 요약

> 정찰 phase 결과 정리. 상세 로그는 [../../doc/09_EHANJA_ONLINE_REVERSING.md](../../doc/09_EHANJA_ONLINE_REVERSING.md).
> 이 문서는 **수집 설계에 직접 참조하는 요약본**.

## 기본 정보

- **URL 베이스**: `http://www.e-hanja.kr/` (HTTP only, HTTPS 미지원)
- **기술 스택**: Classic ASP + jQuery 3.7.1 AJAX
- **Frame 구조**: 루트가 frameset → `/dic/dictionary.asp`를 메인 프레임에 로드
- **인코딩**: UTF-8 (URL 인코딩 `%E9%91%91` = 鑑)
- **총 커버리지**: **71,716자** (유니코드 13.0, E/F 제외; 鑑 기준 확인)
- **robots.txt**: 없음 (메인으로 리다이렉트 — 명시적 금지 부재)

## 데이터 서브도메인 3개

### ① `www.e-hanja.kr` — UI·상세 페이지 (**세션 필요**)

- 페이지 shell + AJAX 조합
- 세션 쿠키: `ASPSESSIONID*` (첫 GET 때 발급)
- AJAX 엔드포인트는 같은 shell과 이름 쌍: `<page>.asp` ↔ `<page>A.asp`
- 주요 엔드포인트:
  - `/dic/contents/jajun_content.asp?hanja=X` — 글자 상세 shell
  - `/dic/contents/jajun_contentA.asp` — 상세 AJAX (POST)
  - `/dic/jajun/jajun_search.asp?keyword=X` — 검색 shell
  - `/dic/jajun/jajun_searchA.asp` — 검색 AJAX (POST)
  - `/dic/contents/pop_jajun[A].asp` — 글자 팝업
  - `/dic/contents/pop_word[A].asp` — 단어 팝업
  - `/dic/word/...` — 단어 사전
  - `/dic/etc/...` — 교차 참조 (한중일, 유의, 약자, 속자, 장단음 등)
  - `/dic/grade/...` — 교육·법적 분류

### ② `img.e-hanja.kr` — 이미지 서버 (**세션 불필요**)

- 순수 정적 GET
- SVG 패턴:
  ```
  http://img.e-hanja.kr/hanjaSvg/aniSVG/{(cp & 0xFF00):X}/{cp:X}.svg
  ```
- SVG 종류:
  - **Animated** (BMP 통합 + Ext A 대부분): 획별 outline path, `stroke-radical`/`stroke-normal` 클래스
  - **Static** (Ext B/C/D/E): monolithic path
- 누적 프레임 `{cp}-NN.svg`도 있지만 composite에서 유도 가능 → 수집 불필요

### ③ `tool.img.e-hanja.kr` — JSON API (**세션 불필요**)

- 순수 POST + JSON body
- 단일 엔드포인트: `/hanjaSvg/asp/dbHandle.comsTree.asp`
- 요청 형식:
  ```
  POST http://tool.img.e-hanja.kr/hanjaSvg/asp/dbHandle.comsTree.asp
  Content-Type: application/json
  Body: {"request": "<type>", "unicode": "U+<hex>", "db": "server"}
  ```
- 지원 request type (3개):
  - `getHunum` → `{hanja, hRead}` (훈음)
  - `getJahae` → `[{hanja, orderA, orderB, meaning, root_snd}, ...]` (뜻 리스트)
  - `getSchoolCom` → 10종 변이 관계 필드 (hDup, bonja, sokja, yakja, goja, waja, simple, kanji, dongja, tongja)

## 필드 관점 정리

### `jajun_contentA.asp` 응답 (HTML 스크랩, 14 블록)

| # | 필드 | 예시 (鑑) |
|--|-----|--------|
| 1 | 훈음 + ⭐ 등급 | "(감) 거울 감 ⭐⭐⭐" |
| 2 | 자해 (뜻 리스트) | 9개 meaning, 음별 순서 |
| 3 | 부수 + 어원 | 金(钅)(쇠금部) + 긴 설명 |
| 4 | 총획수 + 부수 획수 | 22 (부수 획수: 8) |
| 5 | 모양자 (decomposition) | "鑑 (金 쇠 금, 성씨 김, 監 볼 감)" |
| 6 | 자원 (etymology) | 형성문자/회의문자 + 해설 |
| 7 | 영문 | "mirror, looking glass." |
| 8 | 한어병음 | (鑑 empty, 다른 글자엔 있음) |
| 9 | 교육용 | 고등용 |
| 10 | 한자검정 급수 | 3급II (2급) |
| 11 | 대법원 인명용 | "인명용" / 없음 |
| 12 | 분류 | (기타 태그) |
| 13 | 동자 (이미지 섬네일) | getSchoolCom과 중복 |
| 14 | 유의 + 관련 복합어 | "거울 경; 鑑賞, 龜鑑, 鑑定, ..." |

### 변이 관계 10종 (`getSchoolCom`)

| 필드 | 한자 | 의미 | 방향 |
|------|-----|-----|-----|
| `bonja` | 本字 | 본자 (원형) | 상위 |
| `sokja` | 俗字 | 속자 (민간) | 하위 |
| `yakja` | 略字 | 약자 (간략) | 하위 |
| `goja` | 古字 | 고자 (옛) | 하위 |
| `waja` | 訛字 | 와자 (오기) | 하위 |
| `simple` | 簡體字 | 간체자 (중국) | 하위 |
| `kanji` | 日本漢字 | 일본 신자체 | 하위 |
| `dongja` | 同字 | 동자 | 대등 |
| `tongja` | 通字 | 통자 | 대등 |
| `hDup` | ? | 용도 불명 | ? |

복합 라벨 가능: `學 ⟷ 学` 한 엣지가 yakja + simple + kanji 동시에 점유.

## 중요 관찰

1. **모바일 DB (10,932자)는 온라인의 순수 subset** — 온라인이 **7배 넓음 + edge 2~3배 촘촘**
2. **SMP Ext B/C/D/E에도 variant 관계 + canonical SVG 존재** (모바일엔 전혀 없음)
3. **`hSchoolCom`의 goja/waja/sokja 필드는 모바일에도 있지만 canonical DB가 미임포트** — 확장 여지
4. **검색 API는 세션+인코딩 민감**하지만 **검색 우회 가능**:
   직접 `jajun_contentA.asp?hanja=X` 요청하면 SVG URL을 포함한 HTML 획득, `tool.img.e-hanja.kr` JSON API는 애초에 세션 불필요.

## 정찰 샘플 저장소

[./samples/](./samples/)
- `9451.svg` 외 — SVG composite 샘플
- `9451-NN.svg` 5개 — 누적 프레임 확인용
- `28C32.svg`, `2B503.svg`, `5ABA.svg` — Ext B/E/A 샘플
- `detailA_鑑.html` — 상세 AJAX 응답 원본
- `pop_jajun_U9451.html`, `pop_word_106866.html` — 팝업 샘플
- `etc_*.html`, `jajun_shapeSrch.html`, `word_search_U9451.html` — 기타 엔드포인트 shell

## 관련 문서

- [COLLECTION_PLAN.md](./COLLECTION_PLAN.md) — 수집 우선순위 + 전략
- [../../doc/09_EHANJA_ONLINE_REVERSING.md](../../doc/09_EHANJA_ONLINE_REVERSING.md) — 정찰 로그 원본
- [../../doc/08_STROKE_SOURCE_COVERAGE_PLAN.md](../../doc/08_STROKE_SOURCE_COVERAGE_PLAN.md) — 상위 motivation
