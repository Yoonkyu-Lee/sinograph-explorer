# 重編國語辭典修訂本 Manual

## Source and Licensing

- **Official title**: `重編國語辭典修訂本`
- **Publisher**: 中華民國教育部
- **Official site**: <https://dict.revised.moe.edu.tw>
- **Download page**: <https://language.moe.gov.tw/001/Upload/Files/site_content/M0001/respub/dict_reviseddict_download.html>
- **License**: CC BY-ND 3.0 Taiwan
- **Local acquisition date**: 2026-04-04
- **Dataset date in local files**: 2025-12-29

이 소스는 `Tongyong Guifan`처럼 단순 문자 목록이 아니라,
대만 교육부의 **대형 중국어 사전 데이터**를 flat workbook 형태로 export한 것이다.

중요한 점은 이 DB가:

- 단자(單字) 항목
- 복사/어휘 항목
- 다음자(多音字) 분기

를 **한 워크북 안에 함께** 담고 있다는 점이다.

즉 이것은 1-character-centric DB라기보다,
**lexical dictionary workbook**에 가깝다.

## File Inventory

로컬 기준 핵심 파일은 두 개다.

- [dict_revised_2015_20251229.xlsx](./MOE_REVISED_DICT/dict_revised_2015_20251229.xlsx)
  - main data workbook
  - 1 worksheet
  - worksheet name: `1141224_辭典匯出`
  - logical sheet range: `A1:R163917`
- [dict_revised_2015_20251229_欄位說明.xlsx](./MOE_REVISED_DICT/dict_revised_2015_20251229_欄位說明.xlsx)
  - field-description workbook
  - 1 worksheet
  - worksheet name: `工作表1`

메인 워크북은 XML 내부 기준으로 `163,917` rows를 가진다.

- row 1: header
- rows 2..163917: actual data

즉 실제 data row 수는 `163,916`이다.

## High-Level Positioning

이 데이터는 다음 두 층을 같이 포함한다.

### Character-like entries

- `字數 = 1`
- 단일 한자 항목
- 이 경우 부수/총획수/부수외획수 같은 문자형 필드가 채워질 가능성이 높다

### Lexical entries

- `字數 >= 2`
- 복사, 성어, 일반 어휘
- 이 경우 문자형 필드보다 pronunciation / gloss / synonym / antonym 쪽이 중요하다

즉 이 워크북은 Sinograph Explorer에서:

- pure character metadata source
- lexical explanation source

중 **후자에 더 가까운 DB**다.

## Workbook Schema

메인 workbook header는 다음 18개 컬럼으로 구성된다.

1. `字詞名`
2. `辭條別名`
3. `字數`
4. `字詞號`
5. `部首字`
6. `總筆畫數`
7. `部首外筆畫數`
8. `多音排序`
9. `注音一式`
10. `變體類型 1:變 2:又音 3:語音 4:讀音`
11. `變體注音`
12. `漢語拼音`
13. `變體漢語拼音`
14. `相似詞`
15. `相反詞`
16. `釋義`
17. `多音參見訊息`
18. `異體字`

필드 설명 workbook은 위 18컬럼의 설명을 제공한다.

## Column-by-Column Description

아래 설명은 `欄位說明.xlsx`의 정의와 로컬 데이터 관찰을 합쳐 정리한 것이다.

### `字詞名`

- 사전 표제어
- character entry에서는 한 글자
- lexical entry에서는 2자 이상 단어/성어

설명 workbook 정의:

- `字詞名稱`

observed non-empty rows:

- `163,916 / 163,916`

### `辭條別名`

- original / alternate title
- 빈 경우가 대부분

설명 workbook 정의:

- `原文名稱`

observed non-empty rows:

- `1,466 / 163,916`

샘플상 외래 고유명사나 일부 특수 항목에서 쓰인다.

### `字數`

- 표제어의 한자 글자 수

설명 workbook 정의:

- `字詞名之中文字數`

observed non-empty rows:

- `163,916 / 163,916`

관찰된 분포 상위:

- `2`: `87,183`
- `4`: `35,683`
- `3`: `21,816`
- `1`: `13,415`

즉 이 DB는 단자 사전이라기보다 **복사 중심 사전 데이터**가 압도적으로 많다.

### `字詞號`

- 단자/복사의 internal code

설명 workbook 정의:

- `單字、複詞的編碼`

observed non-empty rows:

- `163,916 / 163,916`

### `部首字`

- 단자 항목의 부수

설명 workbook 정의:

- `單字的所屬部首`

observed non-empty rows:

- `13,415 / 163,916`

즉 사실상 **single-character rows 전용 필드**로 봐도 된다.

### `總筆畫數`

- 단자 항목의 총획수

설명 workbook 정의:

- `單字的總筆畫數`

observed non-empty rows:

- `13,415 / 163,916`

### `部首外筆畫數`

- 단자 항목의 부수 외 획수

설명 workbook 정의:

- `單字的部首外筆畫數`

observed non-empty rows:

- `13,415 / 163,916`

### `多音排序`

- 다음자 분기 index

설명 workbook 정의:

- `0表示非多音字詞，1~6表示本條為多音字詞及本條對應之多音序號。1即(一)、2即(二)，依序以此類推`

observed non-empty rows:

- `163,916 / 163,916`

즉 모든 row에 존재하며, non-polyphonic entries는 `0`으로 표시된다.

### `注音一式`

- 주음 표기

설명 workbook 정의:

- `字詞的注音`

observed non-empty rows:

- `162,434 / 163,916`

### `變體類型 1:變 2:又音 3:語音 4:讀音`

- 변독/또는음/어음/독음 타입 코드

설명 workbook 정의:

- `字詞的音讀變讀及又音、語音、讀音標示`

observed logically non-empty rows:

- `1,606 / 163,916`

주의:

- 많은 row에 공백 문자열이 들어 있으므로,
- XML 차원에서 셀이 있다고 해서 논리적으로 값이 있는 것은 아니다

### `變體注音`

- 변독/또는음 쪽의 주음

설명 workbook 정의:

- `字詞的變讀注音或又音注音、語音注音、讀音注音`

observed non-empty rows:

- `978 / 163,916`

### `漢語拼音`

- 표제어의 한어병음

설명 workbook 정의:

- `字詞的漢語拼音`

observed non-empty rows:

- `162,434 / 163,916`

### `變體漢語拼音`

- 변독/또는음 쪽의 한어병음

설명 workbook 정의:

- `字詞的變讀漢語拼音或又音漢語拼音、語音漢語拼音、讀音漢語拼音`

observed non-empty rows:

- `978 / 163,916`

### `相似詞`

- 유의어

설명 workbook 정의:

- `複詞的相似詞`

observed non-empty rows:

- `13,834 / 163,916`

### `相反詞`

- 반의어

설명 workbook 정의:

- `複詞的相反詞`

observed non-empty rows:

- `8,564 / 163,916`

### `釋義`

- 표제어의 정의/뜻풀이

설명 workbook 정의:

- `字詞的釋義`

observed non-empty rows:

- `163,916 / 163,916`

즉 이 워크북의 핵심 payload는 결국 이 필드다.

### `多音參見訊息`

- 같은 표제어의 다른 음독 항목을 가리키는 cross-reference

설명 workbook 정의:

- `本條其他多音訊息`

observed non-empty rows:

- `5,046 / 163,916`

예:

- `(二)ㄆㄚˊ pá（00584）`

### `異體字`

- 이 row가 이체자 성격을 갖는지 알려주는 표시

설명 workbook 정의:

- `表該單字為異體字`

observed non-empty rows:

- `1,482 / 163,916`

샘플상 값은 주로 `異體字`라는 flag-like text다.

## Structural Interpretation

이 워크북은 정규화된 관계형 DB가 아니라,
**평탄화된 사전 export sheet**로 이해하는 것이 맞다.

즉 한 row가 한 dictionary sense block에 대응하고,
필요하면 다음 정보가 같이 붙는다.

- 표제어
- 발음
- 병음
- 뜻풀이
- 유의/반의
- 다음자 참조
- 이체자 flag

그리고 단자 행에 한해서:

- 부수
- 총획수
- 부수외획수

가 추가된다.

## Coverage / Limits

강점:

- 매우 큰 규모의 공개 중국어 사전 데이터
- 단자와 복사를 한 소스에서 함께 다룸
- 주음/한어병음/뜻풀이가 풍부함
- 다음자 분기와 참조 정보가 존재함

제약:

- pure character DB가 아님
- variant family graph를 구조적으로 제공하지는 않음
- 단자 필드와 복사 필드가 같은 sheet에 섞여 있어 후처리가 필요
- 라이선스가 `CC BY-ND`라서, 가공/재배포 정책은 주의 깊게 해석해야 함

## Recommended Integration Role

Sinograph Explorer에서는 이 소스를 다음처럼 쓰는 것이 자연스럽다.

### Primary role

- Chinese lexical / explanatory layer
- Zhuyin + pinyin dictionary support
- polyphonic cross-reference support
- single-character rows에 대한 traditional Chinese gloss 보강

### Suggested canonical attachment

```text
character_entry
  character
  moe_revised
    pronunciation_zhuyin
    pronunciation_pinyin
    radical
    total_strokes
    residual_strokes
    definitions
    polyphonic_index
    polyphonic_crossrefs
    variant_flag

lexical_entry
  headword
  headword_len
  bopomofo
  pinyin
  synonyms
  antonyms
  definitions
```

### Not primary role

- multilingual Japanese/Korean backbone
- Unicode-wide rare-character coverage backbone
- explicit variant-family authority

즉 이 소스는 `Unihan`이나 `異體字` 계열을 대체하지 않고,
그 위에 **중국어권 정의와 주음/병음 설명을 보강하는 대형 lexical source**로 배치하는 것이 가장 적절하다.
