# MakeMeAHanzi Manual

이 문서는 [dictionary.txt](./MAKEMEAHANZI/dictionary.txt)와 [graphics.txt](./MAKEMEAHANZI/graphics.txt)의 구조를 정리한 매뉴얼이다.  
MakeMeAHanzi는 Unihan이나 e-hanja처럼 범용 다국어 한자사전 DB가 아니라, **한자 분해(decomposition)** 와 **획순/획 형상(graphics)** 에 특화된 compact dataset으로 이해하는 것이 가장 정확하다.

## Source And Licensing

- **프로젝트**: Make Me a Hanzi
- **GitHub**: https://github.com/skishore/makemeahanzi
- **수집일**: 2026-04-04
- **로컬 보관 위치**: [MAKEMEAHANZI](./MAKEMEAHANZI)

### License Caveat

이 소스는 단일 라이선스형 DB라기보다, 여러 출처를 조합한 dataset에 가깝다.

- 사전 계열 데이터는 HanDeDict 계열 출처가 얽혀 있을 수 있음
- graphics/stroke 계열은 Wikimedia SVG 등 외부 자산 계열의 영향이 있을 수 있음
- 따라서 **파일 단위와 자산 단위의 provenance를 따로 보는 것이 안전**하다

실제 재배포/학습데이터 생성/앱 탑재 단계에서는 upstream `COPYING`과 원출처 표기를 다시 확인해야 한다.

## File Inventory

이 DB는 사실상 두 개의 JSON Lines 파일로 구성된다.

- [dictionary.txt](./MAKEMEAHANZI/dictionary.txt)
  - JSON Lines
  - **9,574 entries**
  - 의미, 병음, 분해식, 어원 설명 등 문자 메타데이터 담당
- [graphics.txt](./MAKEMEAHANZI/graphics.txt)
  - JSON Lines
  - **9,574 entries**
  - 획 SVG path와 stroke median 데이터 담당

즉 구조적으로는:

1. `dictionary.txt` = semantic / structural layer
2. `graphics.txt` = drawing / trajectory layer

라고 보면 된다.

## High-Level Positioning

이 소스의 성격은 아래처럼 정리할 수 있다.

- **not** a general-purpose multilingual sinograph dictionary
- **not** a variant-family authority
- **is** a character decomposition dataset
- **is** a stroke graphics / stroke-order support dataset
- **is** a useful seed source for synthetic sample generation

즉 Sinograph Explorer 전체 스택 안에서는:

- Unihan = variant / codepoint backbone
- e-hanja = Korean explanation / Korean hun-eum richness
- KANJIDIC2 = Japanese reading / dictionary refs
- MakeMeAHanzi = decomposition + graphics

라는 역할 분담이 자연스럽다.

## `dictionary.txt`

### File Shape

`dictionary.txt`는 **one JSON object per line** 형식이며, 실질적인 primary key는 `character`이다.

예상 접근 방식:

```python
for line in dictionary_txt:
    record = json.loads(line)
    key = record["character"]
```

### Observed Top-Level Fields

로컬 파일 스캔 기준으로 관찰된 top-level field는 아래 7개다.

- `character`
- `definition`
- `pinyin`
- `decomposition`
- `radical`
- `matches`
- `etymology`

관찰 빈도:

- `character`: 9,574 / 9,574
- `decomposition`: 9,574 / 9,574
- `pinyin`: 9,574 / 9,574
- `radical`: 9,574 / 9,574
- `matches`: 9,574 / 9,574
- `definition`: 9,516 / 9,574
- `etymology`: 9,033 / 9,574

즉 `definition`과 `etymology`는 optional이고, 나머지는 사실상 항상 존재한다.

### Field Descriptions

#### `character`

- 단일 한자 문자
- 이 소스에서 가장 실용적인 join key
- 다른 DB와 통합할 때도 우선 `character` -> `codepoint`로 연결하는 것이 자연스럽다

#### `definition`

- 영어 gloss
- optional
- **58 entries**에서 누락
- 매우 짧은 의미 요약인 경우가 많고, 장문의 사전 정의를 기대하면 안 된다

예:

- `learning, knowledge, science; to study, to go to school; -ology`

#### `pinyin`

- pinyin 문자열의 list
- 대부분 1개 항목을 갖지만 list 형태로 저장됨
- 빈 list일 수 있음
- **9 entries**에서 empty list

즉 단일 string으로 취급하지 말고, 항상 list로 파싱하는 게 안전하다.

#### `decomposition`

- IDS-like decomposition string
- `⿰`, `⿱`, `⿲`, `⿳`, `⿴`, `⿵`, `⿶`, `⿷`, `⿸`, `⿹`, `⿺`, `⿻` 같은 IDC 연산자를 사용
- 문자 구조를 “좌우”, “상하”, “삼분 구조” 등으로 분해한 표현

현재 로컬 UTF-8 스캔에서는 literal `？` placeholder는 검출되지 않았고, decomposition은 IDC operator 기반 문자열로 구성되어 있었다.  
다만 PowerShell/콘솔 인코딩이 깨질 경우 placeholder처럼 보이는 mojibake가 섞여 보일 수 있으므로, 파일 자체는 UTF-8 기준으로 읽는 것이 중요하다.

#### `radical`

- 이 데이터셋이 해당 문자의 부수/대표 radical로 보는 문자
- 숫자 ID가 아니라 문자 자체로 저장됨
- 예: `子`, `日`, `方`

#### `matches`

- decomposition component와 glyph stroke group 사이의 alignment/mapping 정보로 보이는 list
- 각 원소는:
  - component index list (`[0]`, `[1]`, `[0, 1]` 등)
  - 또는 `null`
- **483 entries**에서 `null`이 포함되어 있어, 이 매핑은 완전하지 않다

보수적으로 해석하면:

- `matches[n]`는 decomposition 쪽 일부 성분이 실제 glyph stroke cluster와 어떻게 대응되는지를 나타내는 보조 alignment 정보
- 하지만 이 필드는 업스트림의 공식 스펙 없이 과해석하지 않는 것이 좋다

즉 현재 프로젝트에서는:

- 분해-획 정렬 힌트
- synthetic rendering 보조 메타데이터

정도로 사용하는 것이 안전하다.

#### `etymology`

- optional object
- **541 entries**에서 누락
- 문자 형성 원리와 설명 힌트를 제공

이 필드는 설명적(explanatory)이고 sparse하다.  
모든 문자에 대해 존재하지 않으며, “엄밀한 고문자학 ground truth”라기보다 학습용/설명용 요약으로 이해하는 편이 안전하다.

### `etymology` Nested Fields

관찰된 nested field는 아래 4개다.

- `type`
- `hint`
- `phonetic`
- `semantic`

관찰 빈도:

- `type`: 9,033
- `hint`: 8,948
- `phonetic`: 6,809
- `semantic`: 6,917

즉 `type`은 `etymology` object가 있으면 거의 항상 있고, `phonetic` / `semantic` / `hint`는 일부만 존재한다.

#### `etymology.type`

관찰된 값은 정확히 아래 3개다.

- `pictophonetic`: **6,966**
- `ideographic`: **1,840**
- `pictographic`: **227**

해석:

- `pictophonetic`
  - 형성자 계열 설명
  - `phonetic`, `semantic`이 같이 붙는 경우가 많음
- `ideographic`
  - 회의/조합적 설명에 가까움
- `pictographic`
  - 상형 설명

#### `etymology.hint`

- 사람이 읽기 쉬운 짧은 설명 문자열
- 학습용 힌트 역할이 강함

#### `etymology.phonetic`

- 형성자 계열에서 음부(phonetic component)를 나타내는 문자

#### `etymology.semantic`

- 형성자 계열에서 의미부(semantic component)를 나타내는 문자

## `graphics.txt`

### File Shape

`graphics.txt`도 **one JSON object per line** 형식이며, `character`를 key로 하는 1자 1레코드 구조다.

현재 로컬 파일 기준으로 `dictionary.txt`와 **동일하게 9,574 entries**를 가지므로, character coverage는 one-to-one로 맞춰진 것으로 보는 것이 자연스럽다.

### Observed Top-Level Fields

관찰된 field는 정확히 아래 3개다.

- `character`
- `strokes`
- `medians`

세 필드 모두 **9,574 / 9,574**로 항상 존재한다.

### Field Descriptions

#### `character`

- 단일 한자 문자
- `dictionary.txt.character`와 대응되는 join key

#### `strokes`

- SVG path string list
- 각 원소가 한 획을 나타냄
- 즉 `len(strokes)`는 stroke count로 해석 가능

이 필드는 실제 렌더링 가능한 벡터 path이므로:

- stroke-order playback
- vector-to-raster rendering
- 학습용 glyph 이미지 생성

에 직접 활용 가능하다.

#### `medians`

- 각 획의 median / centerline을 나타내는 polyline point list
- 보통 `[[x1, y1], [x2, y2], ...]` 형태의 점열
- 한 획의 “중심 경로”를 제공하는 보조 데이터

### Positional Alignment

현재 구조상 보수적인 기본 가정은 다음과 같다.

- `strokes[n]` and `medians[n]` refer to the same stroke
- 즉 `strokes`와 `medians`는 positional alignment를 가진다

이 가정은 파일 구조와 샘플에서 자연스럽지만, 공식 upstream schema를 더 확인하기 전까지는 “observed alignment”로 표현하는 것이 안전하다.

### Intended Interpretation

`graphics.txt`는 이 데이터셋의 graphics/trajectory side다.

즉 이 파일은:

- 단순 획수 표가 아니라
- 실제 SVG path와 median trajectory를 동시에 담는
- rendering-friendly stroke graphics dataset

으로 이해하면 된다.

## Coverage / Limits

이 소스의 한계는 꽤 명확하다.

### Strengths

- decomposition 정보가 매우 잘 정리되어 있음
- stroke SVG와 median 데이터가 동시에 있음
- JSON Lines라 파싱이 단순함
- synthetic sample generation의 seed source로 좋음

### Limits

- **total entries: 9,574**
  - Unihan, CNS11643 같은 대형 코드포인트 backbone보다 훨씬 작음
- full long-tail Unicode coverage가 아님
  - common / learner-oriented character coverage 쪽에 가까움
- multilingual dictionary로는 약함
  - Korean reading 없음
  - Japanese reading 없음
  - English definition + pinyin 위주
- variant family authority가 아님
  - 간번/정체/semantic variant 그래프를 주력으로 제공하지 않음
- sparse field 존재
  - `definition` missing: 58
  - `etymology` missing: 541
  - empty `pinyin`: 9
  - `matches`에 `null` 포함: 483

즉 이 DB 하나만으로 “Sinograph Explorer 사전층”을 만들면 부족하고, 반드시 다른 source와 결합해야 한다.

## Integration Notes For Sinograph Explorer

이 소스의 권장 merge position은 아래와 같다.

### Primary Role

- decomposition backbone
- stroke graphics source
- synthetic sample generation seed

### Not Primary Role

- multilingual dictionary backbone
- variant-family authority
- Korean/Japanese reading source

### Recommended Canonical Merge

통합 DB에서는 다음처럼 source-specific extension으로 붙이는 것이 자연스럽다.

```text
character
codepoint

makemeahanzi
  decomposition
  radical
  definition_en
  pinyin
  etymology
    type
    hint
    phonetic
    semantic
  graphics
    strokes
    medians
  matches
```

즉 canonical key는:

- `character`
- 필요하면 `ord(character)` 기반 `codepoint`

로 두고, MakeMeAHanzi 쪽 정보는 extension namespace처럼 부착하는 게 좋다.

### Complementarity

MakeMeAHanzi는 아래 DB들과 상호보완 관계에 있다.

- **Unihan**
  - variants / codepoint backbone / IRG / 일부 readings
- **e-hanja**
  - Korean explanation / Korean hun-eum / Korean educational framing
- **KANJIDIC2**
  - Japanese on/kun readings / Japanese dictionary references

요약하면:

- Unihan = who is this character in Unicode?
- e-hanja = how does Korea explain and classify this character?
- KANJIDIC2 = how does Japanese kanji lexicography describe it?
- MakeMeAHanzi = how is it decomposed and drawn?

## Practical Use In The OCR Project

이 소스는 Lab 3 OCR/Explorer 계획에서 특히 아래 용도로 중요하다.

1. **stroke-based synthetic rendering**
   - SVG path 기반 렌더링
2. **decomposition-aware augmentation**
   - component-level reasoning 보조
3. **structure-aware uncommon character analysis**
   - 희귀자라도 구조가 비슷한 군을 찾는 힌트
4. **future model features**
   - radical/component auxiliary targets
   - decomposition-informed classifier or retrieval model

즉 MakeMeAHanzi는 “사전 DB”라기보다, **구조/획/렌더링 엔진용 DB**로 보는 게 가장 맞다.
