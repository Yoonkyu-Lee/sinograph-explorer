# KANJIDIC2 Manual

이 문서는 [kanjidic2.xml](./KANJIDIC2/KANJIDIC2_xml/kanjidic2.xml)의 구조를 빠르게 파악하기 위한 매뉴얼이다.

핵심 성격:
- KANJIDIC2는 `e-hanja`처럼 여러 테이블을 join하는 관계형 DB가 아니다.
- KANJIDIC2는 `Unihan`처럼 `U+XXXX    kField    value`가 여러 파일에 흩어진 속성 DB도 아니다.
- 대신 **한 글자 = 한 `<character>` XML 엔트리**에 정보가 묶여 있는 **엔트리형 XML 사전 DB**다.

현재 내려받은 파일 기준 헤더:
- `file_version = 4`
- `database_version = 2026-095`
- `date_of_creation = 2026-04-05`

---

## 1. 최상위 구조

```xml
<kanjidic2>
  <header>
    <file_version>...</file_version>
    <database_version>...</database_version>
    <date_of_creation>...</date_of_creation>
  </header>

  <character>
    ...
  </character>
  <character>
    ...
  </character>
  ...
</kanjidic2>
```

DTD 기준 최상위 정의:

```xml
<!ELEMENT kanjidic2 (header,character*)>
<!ELEMENT header (file_version,database_version,date_of_creation)>
<!ELEMENT character (literal,codepoint, radical, misc, dic_number?, query_code?, reading_meaning?)>
```

즉:
- `header`는 파일 자체 버전 정보
- `character`는 한 글자 엔트리
- `character`는 필수 필드 4개와 선택 필드 3개로 구성된다.

---

## 2. `<character>` 엔트리 구조

한 글자 엔트리는 아래 순서로 들어간다.

1. `literal`
2. `codepoint`
3. `radical`
4. `misc`
5. `dic_number` (optional)
6. `query_code` (optional)
7. `reading_meaning` (optional)

트리로 보면:

```text
character
├─ literal
├─ codepoint
│  └─ cp_value+
├─ radical
│  └─ rad_value+
├─ misc
│  ├─ grade?
│  ├─ stroke_count+
│  ├─ variant*
│  ├─ freq?
│  ├─ rad_name*
│  └─ jlpt?
├─ dic_number?
│  └─ dic_ref+
├─ query_code?
│  └─ q_code+
└─ reading_meaning?
   ├─ rmgroup*
   │  ├─ reading*
   │  └─ meaning*
   └─ nanori*
```

---

## 3. 필드별 설명

### `literal`
- 정의: 문자 자체
- 타입: plain text
- 예:

```xml
<literal>斈</literal>
```

실질적으로는 이 값이 사용자 lookup의 기준 key가 된다.

---

### `codepoint`
- 정의: 이 글자가 여러 문자셋/표준에서 어떤 코드값을 가지는지
- 구조:

```xml
<codepoint>
  <cp_value cp_type="...">...</cp_value>
  ...
</codepoint>
```

#### `cp_value`
- 값: 실제 코드
- 속성: `cp_type`

DTD에 나온 `cp_type` 값:
- `jis208`
- `jis212`
- `jis213`
- `ucs`

의미:
- `ucs`: Unicode codepoint
- `jis208`, `jis212`, `jis213`: 일본 JIS 표준 문자셋 코드

예:

```xml
<codepoint>
  <cp_value cp_type="ucs">6588</cp_value>
  <cp_value cp_type="jis208">1-53-61</cp_value>
</codepoint>
```

---

### `radical`
- 정의: 부수 정보
- 구조:

```xml
<radical>
  <rad_value rad_type="...">...</rad_value>
</radical>
```

#### `rad_value`
- 값: 부수 번호
- 속성: `rad_type`

DTD에 나온 `rad_type` 값:
- `classical`
- `nelson_c`

의미:
- `classical`: 강희자전 계열의 고전 부수 체계
- `nelson_c`: Classic Nelson 사전 부수 체계

예:

```xml
<radical>
  <rad_value rad_type="classical">67</rad_value>
</radical>
```

---

### `misc`
- 정의: 한자의 일반 메타데이터
- 구조:

```xml
<misc>
  <grade>...</grade>?
  <stroke_count>...</stroke_count>+
  <variant ...>...</variant>*
  <freq>...</freq>?
  <rad_name>...</rad_name>*
  <jlpt>...</jlpt>?
</misc>
```

#### `grade`
- 일본 교육용/상용 한자 급수
- DTD 설명:
  - `1~6`: 초등학교 학년별 교육용 한자
  - `8`: 중학교에서 배우는 나머지 Jouyou kanji
  - `9`: 인명용 한자
  - `10`: Jouyou variant인 인명용 한자

#### `stroke_count`
- 총획수
- 하나 이상 들어갈 수 있음
- 첫 번째가 대표값이고, 뒤의 값은 흔한 오획수일 수 있음

#### `variant`
- 관련 변체자/대체 표기 또는 대체 인덱스 코드
- 구조:

```xml
<variant var_type="...">...</variant>
```

DTD에 나온 `var_type` 값:
- `jis208`
- `jis212`
- `jis213`
- `deroo`
- `njecd`
- `s_h`
- `nelson_c`
- `oneill`
- `ucs`

중요:
- KANJIDIC2의 variant는 **Unihan처럼 `traditional / simplified / semantic`을 직접 말해주지 않는다.**
- 대신 “다른 표준/사전에서 대응되는 변이 코드”를 넣는 방식이다.
- 따라서 variant를 사람이 읽기 쉬운 문자 관계로 바꾸려면 후처리가 필요하다.

#### `freq`
- 현대 일본어 사용 빈도 순위
- 범위는 대략 `1~2500`
- 값이 없으면 빈도 순위 밖

#### `rad_name`
- 그 글자 자체가 부수일 때의 부수 이름
- 히라가나로 들어감

#### `jlpt`
- 예전 JLPT 체계에서의 급수
- `1~4`
- 신 JLPT N1~N5 체계 전환 이전 데이터

---

### `dic_number`
- 정의: 여러 사전/참고서에서의 인덱스 번호
- 구조:

```xml
<dic_number>
  <dic_ref dr_type="...">...</dic_ref>
  ...
</dic_number>
```

#### `dic_ref`
- 값: 사전 인덱스 번호
- 속성:
  - `dr_type` (required)
  - `m_vol` (optional)
  - `m_page` (optional)

DTD에 나온 `dr_type` 값:
- `nelson_c`
- `nelson_n`
- `halpern_njecd`
- `halpern_kkd`
- `halpern_kkld`
- `halpern_kkld_2ed`
- `heisig`
- `heisig6`
- `gakken`
- `oneill_names`
- `oneill_kk`
- `moro`
- `henshall`
- `sh_kk`
- `sh_kk2`
- `sakade`
- `jf_cards`
- `henshall3`
- `tutt_cards`
- `crowley`
- `kanji_in_context`
- `busy_people`
- `kodansha_compact`
- `maniette`

특이점:
- `moro`는 `m_vol`, `m_page`가 추가될 수 있다.

예:

```xml
<dic_number>
  <dic_ref dr_type="nelson_c">2065</dic_ref>
  <dic_ref dr_type="moro" m_vol="5" m_page="0602">13453</dic_ref>
</dic_number>
```

---

### `query_code`
- 정의: 글자 모양 기반 검색 코드
- 구조:

```xml
<query_code>
  <q_code qc_type="...">...</q_code>
  ...
</query_code>
```

#### `q_code`
- 값: 실제 검색 코드
- 속성:
  - `qc_type` (required)
  - `skip_misclass` (optional)

DTD에 나온 `qc_type` 값:
- `skip`
- `sh_desc`
- `four_corner`
- `deroo`
- `misclass`

의미:
- `skip`: Halpern의 SKIP 코드
- `sh_desc`: Spahn & Hadamitzky descriptor
- `four_corner`: 사각번호법
- `deroo`: De Roo 코드
- `misclass`: 오분류/오인식 가능 코드

`skip_misclass` 값:
- `posn`
- `stroke_count`
- `stroke_and_posn`
- `stroke_diff`

예:

```xml
<query_code>
  <q_code qc_type="skip">2-2-5</q_code>
  <q_code qc_type="sh_desc">3n4.2</q_code>
  <q_code qc_type="four_corner">0040.7</q_code>
</query_code>
```

---

### `reading_meaning`
- 정의: 여러 언어의 읽기와 뜻
- 구조:

```xml
<reading_meaning>
  <rmgroup>
    <reading ...>...</reading>*
    <meaning ...>...</meaning>*
  </rmgroup>*
  <nanori>...</nanori>*
</reading_meaning>
```

#### `rmgroup`
- 읽기와 뜻을 한 묶음으로 다루는 그룹
- 장기적으로는 “읽기별로 다른 뜻”을 묶기 위한 구조

#### `reading`
- 읽기/발음
- 속성:
  - `r_type` (required)
  - `on_type` (optional)
  - `r_status` (optional)

DTD에 나온 `r_type` 값:
- `pinyin`
- `korean_r`
- `korean_h`
- `vietnam`
- `ja_on`
- `ja_kun`

의미:
- `pinyin`: 중국어 병음
- `korean_r`: 한국어 독음 로마자
- `korean_h`: 한국어 독음 한글
- `vietnam`: 베트남식 한자음
- `ja_on`: 일본 음독
- `ja_kun`: 일본 훈독

`on_type`:
- DTD 설명상 `kan`, `go`, `tou`, `kan'you` 같은 음독 타입이 들어갈 수 있으나, 현재는 거의 안 쓰인다고 적혀 있다.

`r_status`:
- DTD 설명상 `jy` 같은 값을 통해 Jouyou approved reading 여부를 표시할 수 있으나, 현재는 거의 안 쓰인다고 적혀 있다.

#### `meaning`
- 뜻
- 속성:
  - `m_lang` (optional)

의미:
- `m_lang`이 없으면 영어(`en`)가 기본
- 있으면 ISO 639-1 두 글자 언어 코드

즉 KANJIDIC2는 영어 뜻을 기본으로 하고, 필요하면 다른 언어 meaning도 넣을 수 있게 설계되어 있다.

#### `nanori`
- 인명에만 남아 있는 일본식 이름 읽기

예:

```xml
<reading_meaning>
  <rmgroup>
    <reading r_type="pinyin">xue2</reading>
    <reading r_type="korean_r">hag</reading>
    <reading r_type="korean_h">학</reading>
    <reading r_type="ja_on">ガク</reading>
    <reading r_type="ja_kun">まな.ぶ</reading>
    <meaning>learning</meaning>
    <meaning>knowledge</meaning>
    <meaning>school</meaning>
  </rmgroup>
  <nanori>さとる</nanori>
</reading_meaning>
```

---

## 4. `斈` 예시로 보면

`斈` 엔트리는 실제로 다음 정보를 갖는다.

- `literal = 斈`
- `cp_value(ucs) = 6588`
- `cp_value(jis208) = 1-53-61`
- `rad_value(classical) = 67`
- `stroke_count = 7`
- `variant = jis208/jis212/nelson_c 코드들`
- `dic_ref = nelson_c, nelson_n, halpern_njecd, halpern_kkd, moro`
- `q_code = skip, sh_desc, four_corner`
- readings:
  - `pinyin = xue2`
  - `korean_r = hag`
  - `korean_h = 학`
  - `vietnam = Học`
  - `ja_on = ガク`
  - `ja_kun = まな.ぶ`
- meanings:
  - `learning`
  - `knowledge`
  - `school`

즉 `斈` 같은 글자도 KANJIDIC2에서는 꽤 정보가 잘 들어 있는 편이다.

---

## 5. Unihan / e-hanja와 비교

### Unihan과 비교
- Unihan은 코드포인트 중심 속성 DB
- KANJIDIC2는 글자 엔트리 중심 XML 사전
- Unihan은 variant 타입이 직접적
- KANJIDIC2는 variant가 표준 코드/사전 번호 기반이라 후처리가 더 필요

### e-hanja와 비교
- e-hanja는 관계형 앱 DB
- KANJIDIC2는 단일 XML 파일
- e-hanja는 한국어 자훈/세부 풀이가 강함
- KANJIDIC2는 일본 쪽 사전 인덱스와 온/훈독, 검색 코드가 강함

---

## 6. 프로젝트에서의 활용 포인트

KANJIDIC2는 특히 다음 용도에 좋다.

- 한자 1자 lookup용 compact entry
- 일본 음독/훈독 보강
- 중국어 병음 / 한국어 독음 보강
- 사전 인덱스 번호 확보
- 부수/획수/검색 코드 확보

반면 다음은 약하다.

- variant 의미 해석
  - `traditional / simplified / semantic` 같은 관계를 직접 주지 않음
- 한국식 자훈
- 한국 사전식 세부 풀이

즉 실제 앱에선 이런 식이 자연스럽다.

- **Variant backbone**: Unihan
- **한국어 설명/자훈**: e-hanja
- **읽기/사전 인덱스/온훈독 보강**: KANJIDIC2

---

## 7. 빠른 요약

- KANJIDIC2는 **한 글자당 하나의 `<character>` 엔트리**로 구성된 XML 사전이다.
- 필수 축은 `literal`, `codepoint`, `radical`, `misc`이다.
- 선택 축은 `dic_number`, `query_code`, `reading_meaning`이다.
- 주요 강점은 **읽기와 사전 인덱스, 검색 코드**다.
- 주요 약점은 **variant 관계의 의미가 직접적이지 않다는 점**이다.
