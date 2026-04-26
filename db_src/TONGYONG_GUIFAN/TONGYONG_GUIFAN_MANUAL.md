# 通用规范汉字表 Manual

## Source and Positioning

- **Official title**: `通用规范汉字表`
- **Publisher**: 中华人民共和国教育部 / 国家语言文字工作委员会
- **Promulgation**: 2013-06
- **Local source file**: [tongyong_guifan_2013.csv](./TONGYONG_GUIFAN/tongyong_guifan_2013.csv)
- **Upstream source chain**: official standard text mirrored through Wikisource and GitHub
- **Local acquisition date**: 2026-04-04

이 데이터는 `Unihan`이나 `KANJIDIC2`처럼 다층 메타데이터를 담은 사전 DB가 아니다.  
성격상 가장 가까운 것은 **중국 본토의 표준 상용 한자 리스트**이며, 각 문자에 대해 “이 문자가 국가 표준 범용자 목록의 몇 급에 속하는가”를 알려주는 **정렬된 기준표**에 가깝다.

즉 이 소스의 주된 역할은:

- 중국 표준 상용자 coverage 기준 제공
- common vs less-common character tiering
- OCR synthetic data generation 시 우선순위/난이도 버킷 제공

반대로 이 소스가 직접 제공하지 않는 것은:

- 뜻풀이
- 병음/독음
- 이체자 관계
- 부수/획수
- 자형 구조

## File Inventory

현재 로컬 기준으로 핵심 파일은 하나다.

- [tongyong_guifan_2013.csv](./TONGYONG_GUIFAN/tongyong_guifan_2013.csv)
  - CSV
  - `8,105` rows
  - columns: `id`, `character`, `bucket`

## High-Level Data Model

이 CSV는 사실상 다음과 같은 단순 테이블이다.

```text
row
  id
  character
  bucket
```

한 행이 한 문자에 대응한다.

- 한 문자당 정확히 1행
- `character` 중복 없음
- 총 `8,105`자

즉 practical primary key는 `character`이지만,  
표 내부의 정렬/표준 순번을 보존하려면 `id`도 함께 유지하는 것이 좋다.

## Column Schema

### `id`

- zero-padded ordinal string
- 예: `0001`, `0205`, `3501`
- 숫자형으로 바꾸면 1부터 시작하는 표 순번처럼 해석할 수 있다

주의:

- `id` 값 자체는 `1..8105` 범위를 커버하지만,
- 현재 CSV 행 순서는 `id` 오름차순과 완전히 일치하지 않는다
- 예를 들어 35번째 행에서 바로 `0205`가 등장한다

즉 이 필드는 **표준 리스트의 원래 순번 레이블**로 쓰고,
행 순서와 동일하다고 가정하지 않는 편이 안전하다.

### `character`

- single Han character
- 이 파일의 실질적인 문자 key
- 총 `8,105`개가 모두 unique

예:

- `一`
- `乙`
- `二`
- `十`

### `bucket`

- 해당 문자가 속하는 표준 버킷 / usage tier
- observed values:
  - `一级字表`
  - `二级字表`
  - `三级字表`

distribution:

- `一级字表`: `3,500`
- `二级字表`: `3,000`
- `三级字表`: `1,605`

boundary by `id` value:

- `一级字表`: `1..3500`
- `二级字表`: `3501..6500`
- `三级字表`: `6501..8105`

즉 `bucket`은 이 문자 집합의 핵심 의미를 담는 필드이며,
이 DB를 사용하는 이유도 대부분 여기에 있다.

## Observed Properties

직접 로컬 CSV를 순회해 확인한 사실:

- total rows: `8,105`
- unique characters: `8,105`
- duplicate characters: `0`
- bucket values: exactly 3 classes

따라서 이 파일은:

- 누락 없는 1문자 1행 구조
- 매우 단순하고 deterministic한 lookup 소스

로 취급해도 무방하다.

## Practical Interpretation

이 표는 사전이라기보다 **중국 표준 문자의 difficulty / commonness tier table**에 가깝다.

대략적으로는 이렇게 이해하면 된다.

- `一级字表`
  - 가장 기본적이고 널리 쓰이는 상용자 층
- `二级字表`
  - 추가 상용자 층
- `三级字表`
  - 표준에 포함되지만 상대적으로 덜 핵심적인 층

정확한 교육적/정책적 해석은 원 표준 문서를 함께 봐야 하지만,
Sinograph Explorer 프로젝트에서는 우선 다음처럼 쓰면 충분하다.

- tier 1 = 가장 common
- tier 2 = common but secondary
- tier 3 = comparatively less common within the standard set

## Coverage / Limits

장점:

- 중국 본토 표준 상용자 범위를 명확히 제공
- 구조가 단순해서 lookup/merge가 쉬움
- OCR 학습 시 common-character baseline을 만들기 좋음

한계:

- 표준 문자 `8,105`자만 포함
- long-tail Unicode coverage는 매우 부족
- pronunciation / meaning / variant / decomposition 정보 없음
- 사전형 DB가 아니라서 단독 사용 가치는 제한적

즉 이 DB는 **coverage classifier**로는 유용하지만,
standalone dictionary backbone으로는 부적합하다.

## Recommended Integration Role

Sinograph Explorer 통합 DB 안에서는 다음 역할이 적절하다.

### Primary role

- 중국 표준 상용자 여부 표시
- commonness tier 부여
- synthetic data sampling 우선순위 제어

### Suggested canonical attachment

```text
character
  codepoint
  source_flags
  tongyong_guifan
    present
    bucket
    ordinal_id
```

### Not primary role

- multilingual readings
- variant-family authority
- detailed dictionary definition
- structural decomposition

즉 이 소스는 `Unihan`, `e-hanja`, `KANJIDIC2`, `MakeMeAHanzi`를 대체하지 않고,  
그 위에 **“중국 표준 상용자 레이어”**를 덧씌우는 보조 소스로 보는 것이 맞다.
