# db_src — 원본 DB Navigation

이 문서는 `db_src/` 아래 각 DB 폴더가 **무엇이고 / 어디에 강하고 / 매뉴얼은 어디에 있는지**를 한 눈에 보기 위한 index 다.

각 DB 폴더에는 자체 `*_MANUAL.md` 가 들어 있고, 그 안에 source/license/file inventory/schema 가 정리돼 있다. 이 README 는 그 매뉴얼들로 들어가기 위한 입구다.

---

## 1. 한 줄 요약 표

| DB | 폴더 | 매뉴얼 | 한 줄 성격 |
|---|---|---|---|
| Unihan | [Unihan/](./Unihan/) | [UNIHAN_MANUAL.md](./Unihan/UNIHAN_MANUAL.md) | Unicode 한자 코드포인트별 속성 (`k*`) backbone DB |
| e-hanja (앱) | [e-hanja/](./e-hanja/) | [EHANJA_MANUAL.md](./e-hanja/EHANJA_MANUAL.md) | e-hanja 모바일 앱 SQLite/CSV — 한국어 훈음·자해·이체자 풍부 |
| e-hanja Online | [e-hanja_online/](./e-hanja_online/) | [EHANJA_ONLINE_MANUAL.md](./e-hanja_online/EHANJA_ONLINE_MANUAL.md) | www.e-hanja.kr 수집 SVG 76k + median stroke JSONL |
| KANJIDIC2 | [KANJIDIC2/](./KANJIDIC2/) | [KANJIDIC2_MANUAL.md](./KANJIDIC2/KANJIDIC2_MANUAL.md) | 일본어 reading + 사전 index. 한 글자 = 한 `<character>` XML |
| KanjiVG | [KanjiVG/](./KanjiVG/) | [KANJIVG_MANUAL.md](./KanjiVG/KANJIVG_MANUAL.md) | 일본식 자형 획순 SVG + median stroke JSONL |
| MakeMeAHanzi | [MAKEMEAHANZI/](./MAKEMEAHANZI/) | [MAKEMEAHANZI_MANUAL.md](./MAKEMEAHANZI/MAKEMEAHANZI_MANUAL.md) | 한자 decomposition + stroke graphics (9,574 entries) |
| CNS11643 | [CNS11643/](./CNS11643/) | [CNS11643_MANUAL.md](./CNS11643/CNS11643_MANUAL.md) | 대만 國發會 全字庫 — 발음·획수·부수·창힐·자형성분 |
| MOE Revised Dict | [MOE_REVISED_DICT/](./MOE_REVISED_DICT/) | [MOE_REVISED_DICT_MANUAL.md](./MOE_REVISED_DICT/MOE_REVISED_DICT_MANUAL.md) | 重編國語辭典修訂本 — 단자+어휘 통합 lexical workbook (163k rows) |
| MOE Variants | [MOE_VARIANTS/](./MOE_VARIANTS/) | [MOE_VARIANTS_MANUAL.md](./MOE_VARIANTS/MOE_VARIANTS_MANUAL.md) | 異體字字典 — 정자/이체자 hierarchical TSV (106k rows) |
| Tongyong Guifan | [TONGYONG_GUIFAN/](./TONGYONG_GUIFAN/) | [TONGYONG_GUIFAN_MANUAL.md](./TONGYONG_GUIFAN/TONGYONG_GUIFAN_MANUAL.md) | 通用规范汉字表 2013 — 중국 표준 상용자 8,105자 + tier |
| BabelStone IDS | [babelstone_ids/](./babelstone_ids/) | [BABELSTONE_IDS_MANUAL.md](./babelstone_ids/BABELSTONE_IDS_MANUAL.md) | Andrew West IDS DB — region flag 보존 multi-alternate |
| CHISE IDS | [chise_ids/](./chise_ids/) | [CHISE_IDS_MANUAL.md](./chise_ids/CHISE_IDS_MANUAL.md) | CHISE IDS — Unicode 17.0 cover, GPLv2 |
| cjkvi-ids | [cjkvi_ids/](./cjkvi_ids/) | [CJKVI_IDS_MANUAL.md](./cjkvi_ids/CJKVI_IDS_MANUAL.md) | CHISE 파생 정리본, GPLv2 |
| Fonts | [fonts/external/](./fonts/external/) | [LICENSES.md](./fonts/external/LICENSES.md) | OFL 1.1 CJK 폰트 자산 (synthetic OCR rasterize 용) |
| CEDICT | [CEDICT/](./CEDICT/) | _(아직 없음)_ | placeholder — 비어 있음 |

---

## 2. 역할별 분류

각 DB 매뉴얼이 일관되게 채택하는 분담 관점이다.

### 2.1 Codepoint / variant backbone
- **[Unihan](./Unihan/UNIHAN_MANUAL.md)** — 코드포인트 ↔ `k*` 속성. 다국가 표준의 척추.

### 2.2 의미 / 사전 — 한국어
- **[e-hanja](./e-hanja/EHANJA_MANUAL.md)** — 한국어 훈음, 자해, 이체자, FTS 가 풍부한 관계형 DB (SQLite/CSV 4-layer).
- **[e-hanja_online](./e-hanja_online/EHANJA_ONLINE_MANUAL.md)** — 같은 사이트의 온라인 SVG 자산. `e-hanja` 와 **수집 경로/coverage 가 다름**. 주로 stroke graphics 용.

### 2.3 의미 / 사전 — 일본어
- **[KANJIDIC2](./KANJIDIC2/KANJIDIC2_MANUAL.md)** — 일본 reading, 사전 index (단일 XML 엔트리형).

### 2.4 의미 / 사전 — 중국어 (대만)
- **[CNS11643](./CNS11643/CNS11643_MANUAL.md)** — 발음·획수·부수·창힐·자형 성분. 코드 매핑 테이블 다수.
- **[MOE_REVISED_DICT](./MOE_REVISED_DICT/MOE_REVISED_DICT_MANUAL.md)** — 단자 + 어휘 함께 담긴 lexical workbook.
- **[MOE_VARIANTS](./MOE_VARIANTS/MOE_VARIANTS_MANUAL.md)** — 정자/이체자 계층 TSV. Unicode 미할당 글자는 image_path 보유.

### 2.5 의미 / 사전 — 중국어 (대륙)
- **[TONGYONG_GUIFAN](./TONGYONG_GUIFAN/TONGYONG_GUIFAN_MANUAL.md)** — 의미·음 정보는 없음. **상용자 coverage / tier 기준표** 로만 쓴다.

### 2.6 자형 분해 (IDS)
세 소스가 거의 동일 schema 로 정제돼 있다. 비교·교차 검증용.
- **[babelstone_ids](./babelstone_ids/BABELSTONE_IDS_MANUAL.md)** — region flag (G/H/J/K/T/V/X) 보존, multi-alternate 충실.
- **[chise_ids](./chise_ids/CHISE_IDS_MANUAL.md)** — Unicode 17.0 까지 cover (Ext A–J). GPLv2.
- **[cjkvi_ids](./cjkvi_ids/CJKVI_IDS_MANUAL.md)** — CHISE 파생 정리본. 릴리스 주기 더 느림 (Ext J 없음). GPLv2.
- 비교 도구: [ids_coverage_compare.py](./ids_coverage_compare.py) — 3 IDS 소스 vs e-hanja/T1/custom universe coverage 비교.

### 2.7 획순 / centerline / graphics
- **[KanjiVG](./KanjiVG/KANJIVG_MANUAL.md)** — 일본식 자형 SVG (11,662 files) + median stroke JSONL (6,446 chars / 79,246 strokes).
- **[MAKEMEAHANZI](./MAKEMEAHANZI/MAKEMEAHANZI_MANUAL.md)** — decomposition + stroke graphics 결합 dataset (9,574 entries).
- **[e-hanja_online](./e-hanja_online/EHANJA_ONLINE_MANUAL.md)** — 한국식 자형 SVG (76k) + medianized JSONL (16,329 chars / 210k strokes).

### 2.8 폰트 자산
- **[fonts/external/](./fonts/external/LICENSES.md)** — OFL 1.1 CJK 폰트 모음. synthetic OCR 학습 이미지 rasterize 전용 (배포 산출물 아님).

---

## 3. 라이선스 한 눈

| DB | License | 외부 배포 시 주의 |
|---|---|---|
| Unihan | Unicode license | UCD 표기 |
| e-hanja (앱) | 명시 없음 (앱 추출) | **내부 사용 한정** |
| e-hanja_online | 명시 없음 (사이트 수집) | **내부 사용 한정** |
| KANJIDIC2 | EDRDG license | attribution 필요 |
| KanjiVG | CC BY-SA 3.0 | attribution + share-alike |
| MakeMeAHanzi | mixed (HanDeDict 등) | 파일 단위 provenance 확인 |
| CNS11643 | 대만 정부 공개 데이터 | 출처 표기 권장 |
| MOE_REVISED_DICT | CC BY-ND 3.0 Taiwan | no derivative |
| MOE_VARIANTS | 명시 없음 (사이트 mirror) | 검토 필요 |
| TONGYONG_GUIFAN | 정부 표준 (PD 추정) | 출처 표기 |
| babelstone_ids | 명시 없음 (Unicode L2/21-161 우려) | **내부 사용 한정** |
| chise_ids | GPLv2 | share-alike |
| cjkvi_ids | GPLv2 | share-alike |
| fonts/external | OFL 1.1 | in-process rasterize 만, 산출물에 미동봉 |

---

## 4. 관련 / 외부 위치 참조

- **수집·정제 작업 공간**: [../db_mining/](../db_mining/) — 각 DB 의 raw crawl, 파서, 정제 스크립트.
- **상위 설계 문서**: [../doc/05_SINOGRAPH_CANONICAL_DB_V1_PLAN.md](../doc/05_SINOGRAPH_CANONICAL_DB_V1_PLAN.md), [../doc/11_SINOGRAPH_CANONICAL_DB_V2_PLAN.md](../doc/11_SINOGRAPH_CANONICAL_DB_V2_PLAN.md), [../doc/17_CANONICAL_V3_PLAN.md](../doc/17_CANONICAL_V3_PLAN.md) — 이 db_src 들이 어떻게 canonical DB 로 통합되는지.
- **Stroke source coverage**: [../doc/08_STROKE_SOURCE_COVERAGE_PLAN.md](../doc/08_STROKE_SOURCE_COVERAGE_PLAN.md) — KanjiVG / MakeMeAHanzi / e-hanja_online 등 stroke 소스 통합 계획.
