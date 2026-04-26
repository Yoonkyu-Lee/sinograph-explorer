# Unihan Manual

이 문서는 [Unihan.zip](./Unihan/Unihan.zip)과 그 압축 해제본을 기준으로, **Unihan DB의 구조, 파일 체계, 레코드 형식, 그리고 현재 로컬 버전에서 실제로 존재하는 모든 필드**를 정리한 매뉴얼이다.

기준 파일:
- [Unihan.zip](./Unihan/Unihan.zip)
- [Unihan_DictionaryIndices.txt](./Unihan/Unihan_txt/Unihan_DictionaryIndices.txt)
- [Unihan_DictionaryLikeData.txt](./Unihan/Unihan_txt/Unihan_DictionaryLikeData.txt)
- [Unihan_IRGSources.txt](./Unihan/Unihan_txt/Unihan_IRGSources.txt)
- [Unihan_NumericValues.txt](./Unihan/Unihan_txt/Unihan_NumericValues.txt)
- [Unihan_OtherMappings.txt](./Unihan/Unihan_txt/Unihan_OtherMappings.txt)
- [Unihan_RadicalStrokeCounts.txt](./Unihan/Unihan_txt/Unihan_RadicalStrokeCounts.txt)
- [Unihan_Readings.txt](./Unihan/Unihan_txt/Unihan_Readings.txt)
- [Unihan_Variants.txt](./Unihan/Unihan_txt/Unihan_Variants.txt)

로컬 버전:
- Unicode Version: **17.0.0**
- 파일 헤더 날짜: **2025-07-24**

참고 문서:
- Unicode UAX #38: https://www.unicode.org/reports/tr38/

---

## 1. Unihan은 어떤 DB인가

Unihan은 한자 계열 문자를 위한 **Unicode 메타데이터 DB**다.

핵심 성격:
- `e-hanja`처럼 앱용 관계형 DB가 아니다.
- `KANJIDIC2`처럼 한 글자당 하나의 XML 엔트리가 있는 사전형 DB도 아니다.
- 대신 **코드포인트별 속성(property) 테이블들의 묶음**이다.

즉 Unihan은 구조적으로:

```text
코드포인트(U+XXXX) -> 여러 k* 속성들
```

의 형태를 가진다.

예:

```text
U+5B78    kDefinition    learning, study
U+5B78    kMandarin      xue2
U+5B78    kJapaneseOn    ガク
U+5B78    kKorean        HAK
U+5B78    kSimplifiedVariant    U+5B66
```

즉 “한 글자 엔트리 블록”이 아니라, **한 속성당 한 줄**로 쪼개진 형식이다.

---

## 2. 레코드 형식

Unihan의 실제 데이터 라인은 대부분 아래 형식이다.

```text
U+CODEPOINT<TAB>FIELD<TAB>VALUE
```

예:

```text
U+3405    kDefinition    (ancient form of 五) five
U+3405    kJapanese      ゴ
U+3405    kMandarin      wǔ
```

의미:
- 첫 번째 열: codepoint key
- 두 번째 열: property name (`k*`)
- 세 번째 열: property value

### 파서 규칙
- 빈 줄은 무시
- `#`로 시작하는 줄은 주석
- 데이터 줄은 탭 3열
- 같은 codepoint에 여러 줄이 누적됨
- 같은 field가 여러 값을 가질 때는 보통 value 내부에서 공백/콤마/특수 syntax로 표현됨

즉 파싱 결과는 보통 이렇게 만들게 된다.

```python
{
  "U+5B78": {
    "kDefinition": "...",
    "kMandarin": "...",
    "kJapaneseOn": "...",
    ...
  }
}
```

---

## 3. 파일 체계

현재 로컬 버전에는 8개의 주요 txt 파일이 있다.

1. `Unihan_DictionaryIndices.txt`
2. `Unihan_DictionaryLikeData.txt`
3. `Unihan_IRGSources.txt`
4. `Unihan_NumericValues.txt`
5. `Unihan_OtherMappings.txt`
6. `Unihan_RadicalStrokeCounts.txt`
7. `Unihan_Readings.txt`
8. `Unihan_Variants.txt`

중요한 점:
- Unicode 공식 문서도 말하듯, **특정 필드가 항상 특정 파일에만 있다고 가정하는 것은 안전하지 않다.**
- 즉 파서는 파일별로 하드코딩하기보다, **모든 Unihan txt를 한 번에 순회하면서 field 이름으로 처리**하는 게 좋다.

그래도 사람 입장에서는 파일 grouping이 이해에 도움이 되므로, 이 문서에서는 **현재 로컬 버전의 grouping**을 기준으로 정리한다.

---

## 4. 파일별 필드 전체 목록

현재 로컬 Unihan에서 확인된 총 field 수는 **99개**다.

### 4.1 `Unihan_DictionaryIndices.txt`
현재 field 수: **19**

- `kHanYu`
  - 《漢語大字典》 위치 인덱스
- `kIRGHanyuDaZidian`
  - IRG가 참조한 《漢語大字典》 계열 소스 코드
- `kIRGKangXi`
  - IRG가 참조한 강희자전 계열 소스 코드
- `kKangXi`
  - 강희자전 인덱스
- `kMorohashi`
  - Morohashi 대한화사전 인덱스
- `kCihaiT`
  - 《辭海》 계열 인덱스
- `kSBGY`
  - 《宋本廣韻》 계열 인덱스
- `kNelson`
  - Nelson 사전 인덱스
- `kCowles`
  - Cowles reference 인덱스
- `kMatthews`
  - Matthews Chinese-English dictionary 계열 인덱스
- `kGSR`
  - Karlgren의 *Grammata Serica Recensa* 계열 인덱스
- `kFennIndex`
  - Fenn dictionary 계열 인덱스
- `kKarlgren`
  - Karlgren reference 인덱스
- `kSMSZD2003Index`
  - 商務新字典(Soengmou San Zidin, 2003) 인덱스
- `kMeyerWempe`
  - Meyer-Wempe reference 인덱스
- `kLau`
  - Lau 계열 reference 인덱스
- `kCheungBauerIndex`
  - Cheung Bauer 홍콩/광동어 reference 인덱스
- `kDaeJaweon`
  - 대자원(大字源/大字典류가 아니라 한국어권 `大字源`/`大字典` 계열이 아니라 여기서는 **대자전(大字典/大字源과 별도 한국어 reference)**로 쓰이는 항목) 인덱스
- `kIRGDaeJaweon`
  - IRG가 참조한 대자전 계열 소스 코드

### 4.2 `Unihan_DictionaryLikeData.txt`
현재 field 수: **12**

- `kCangjie`
  - 창힐 입력 코드
- `kMojiJoho`
  - 일본 문자정보 기반 reference/code
- `kStrange`
  - unusual / strange character note
- `kPhonetic`
  - 형성자/음부 관련 정보
- `kFenn`
  - Fenn dictionary 계열 데이터
- `kUnihanCore2020`
  - Unihan Core 2020 핵심 집합 여부
- `kFourCornerCode`
  - 사각호마(四角號碼) 코드
- `kCheungBauer`
  - Cheung Bauer 계열 광동어/홍콩 참고 데이터
- `kAlternateTotalStrokes`
  - 대체 총획수
- `kGradeLevel`
  - 교육 등급/학년 계열 분류값
- `kHDZRadBreak`
  - 《漢語大字典》의 부수 분해/분절 관련 정보
- `kHKGlyph`
  - 홍콩 표준 자형 관련 정보

### 4.3 `Unihan_IRGSources.txt`
현재 field 수: **15**

- `kIRG_GSource`
  - 중국(G) 소스
- `kIRG_JSource`
  - 일본(J) 소스
- `kIRG_TSource`
  - 대만(T) 소스
- `kRSUnicode`
  - Unicode 부수-획수 값
- `kTotalStrokes`
  - 총획수
- `kIRG_KSource`
  - 한국(K) 소스
- `kIRG_KPSource`
  - 북한(KP) 소스
- `kIRG_VSource`
  - 베트남(V) 소스
- `kIRG_HSource`
  - 홍콩(H) 소스
- `kIRG_USource`
  - UTC/U-source
- `kIICore`
  - IICore core set membership
- `kIRG_MSource`
  - 마카오(M) 소스
- `kIRG_UKSource`
  - UK source
- `kCompatibilityVariant`
  - Unicode compatibility ideograph와의 대응
- `kIRG_SSource`
  - 싱가포르(S) 소스

### 4.4 `Unihan_NumericValues.txt`
현재 field 수: **6**

- `kOtherNumeric`
  - 기타 숫자값
- `kVietnameseNumeric`
  - 베트남식 숫자값
- `kZhuangNumeric`
  - 장어(Zhuang) 숫자값
- `kPrimaryNumeric`
  - 기본 숫자값
- `kAccountingNumeric`
  - 회계/대자 숫자값
- `kTayNumeric`
  - Tay 언어권 숫자값

### 4.5 `Unihan_OtherMappings.txt`
현재 field 수: **23**

- `kJIS0213`
  - JIS X 0213 mapping
- `kGB3`
  - GB 계열 mapping
- `kTGH`
  - 대만/중화권 표준 계열 mapping field
- `kKoreanName`
  - 한국어 이름 정보
- `kEACC`
  - EACC mapping
- `kTaiwanTelegraph`
  - 대만 전보 코드
- `kBigFive`
  - Big5 mapping
- `kCCCII`
  - CCCII mapping
- `kCNS1986`
  - CNS 1986 mapping
- `kCNS1992`
  - CNS 1992 mapping
- `kGB0`
  - GB 0 mapping
- `kGB1`
  - GB 1 mapping
- `kJis0`
  - JIS 0 mapping
- `kJoyoKanji`
  - 일본 상용한자 분류
- `kKoreanEducationHanja`
  - 한국 교육용 한자 분류
- `kMainlandTelegraph`
  - 중국 대륙 전보 코드
- `kXerox`
  - Xerox mapping
- `kGB5`
  - GB 5 mapping
- `kJis1`
  - JIS 1 mapping
- `kPseudoGB1`
  - pseudo GB1 mapping
- `kGB8`
  - GB 8 mapping
- `kJinmeiyoKanji`
  - 일본 인명용 한자 분류
- `kIBMJapan`
  - IBM Japan mapping

### 4.6 `Unihan_RadicalStrokeCounts.txt`
현재 field 수: **1**

- `kRSAdobe_Japan1_6`
  - Adobe-Japan1-6 기준 radical-stroke 정보

### 4.7 `Unihan_Readings.txt`
현재 field 수: **17**

- `kCantonese`
  - 광동어 읽기
- `kDefinition`
  - 영어 중심 정의/뜻풀이
- `kJapanese`
  - 일본어 읽기 통합 필드
  - `kJapaneseOn`, `kJapaneseKun`보다 예전/통합적인 성격
- `kMandarin`
  - 중국어 표준독음
- `kFanqie`
  - 반절(反切) 정보
- `kHanyuPinyin`
  - 《漢語大字典》 계열 병음 정보
- `kTGHZ2013`
  - 통용규범한자표(2013) 계열 정보
- `kXHC1983`
  - 《現代漢語詞典》 1983 계열 정보
- `kVietnamese`
  - 베트남 한자음
- `kSMSZD2003Readings`
  - 商務新字典(2003) 기반 Mandarin/Cantonese reading
- `kHangul`
  - 한글 표기형
- `kTang`
  - 당음/중세음 계열 정보
- `kJapaneseKun`
  - 일본 훈독
- `kJapaneseOn`
  - 일본 음독
- `kHanyuPinlu`
  - 漢語頻率字典 계열 병음/빈도 정보
- `kKorean`
  - 한국 한자음 로마자
- `kZhuang`
  - 장어권 읽기

### 4.8 `Unihan_Variants.txt`
현재 field 수: **6**

- `kSemanticVariant`
  - 의미상 같은 계열의 변이
- `kSpoofingVariant`
  - 시각적으로 헷갈리는 스푸핑성 변이
- `kTraditionalVariant`
  - 정체 대응 문자
- `kSimplifiedVariant`
  - 간체 대응 문자
- `kSpecializedSemanticVariant`
  - 특정 문맥에서만 의미가 같은 specialized semantic variant
- `kZVariant`
  - 자형 차이 중심의 Z-variant

---

## 5. Unihan의 핵심 체계

### 5.1 codepoint 중심
Unihan에서 모든 것은 결국 `U+XXXX` key에 매달린다.

예:

```text
U+6588    kMandarin    xué
U+6588    kKorean      HAK
U+6588    kSemanticVariant    U+5B78
```

즉 한자 자체는 codepoint가 ID다.

### 5.2 field는 sparse하다
모든 codepoint가 모든 field를 가지는 것은 아니다.

예:
- 어떤 글자는 `kDefinition`은 있지만 `kKorean`은 없음
- 어떤 글자는 `kTraditionalVariant`는 있지만 `kMandarin`은 없음
- 어떤 희귀자는 source 정보만 있고 reading은 거의 비어 있음

즉 Unihan은 **완결 사전**이 아니라 **희소한 메타데이터 집합**이다.

### 5.3 variant는 강점이지만 완결적이지는 않다
Unihan의 장점 중 하나는:
- `kTraditionalVariant`
- `kSimplifiedVariant`
- `kSemanticVariant`
- `kSpecializedSemanticVariant`
- `kSpoofingVariant`
- `kZVariant`

처럼 variant 관계가 명시적으로 있다는 점이다.

그러나:
- 모든 관계가 완벽하지는 않다
- variant family 전체에 뜻/독음이 다 채워진 것도 아니다

즉 **variant graph backbone**으로는 좋지만, 그것만으로 사전이 완성되진 않는다.

### 5.4 file grouping은 편의용일 뿐
현재는 field들이 파일별로 나뉘어 있지만,
파서 관점에서는:
- `Unihan_*.txt` 전부를 읽고
- `field name` 기준으로 처리해야 한다.

이는 Unicode 공식 문서의 권고와도 맞는다.

---

## 6. 프로젝트 관점에서 Unihan의 역할

Sinograph Explorer 프로젝트에서 Unihan은 특히 다음 용도에 적합하다.

- codepoint backbone
- Unicode 존재 여부 확인
- variant graph 구성
- radical / stroke / IRG source 확인
- 일부 다국어 reading 보강
- dictionary index 보강

반면 약한 부분:
- 한국어 자훈
- 한국어 뜻풀이
- 엔트리 completeness
- long-tail rare character에 대한 풍부한 human-readable 설명

즉 현실적으로는:

- **Variant backbone**: Unihan
- **한국어 설명/자훈**: e-hanja
- **일본/다국어 reading + dictionary refs**: KANJIDIC2

식으로 결합하는 것이 자연스럽다.

---

## 7. 추천 파싱 전략

### 최소 파서
1. `Unihan_*.txt` 전부 순회
2. 주석/빈 줄 제거
3. `codepoint`, `field`, `value` 3열 분리
4. `db[codepoint][field] = value`

### 추천 저장 구조

```python
{
  "U+5B78": {
    "char": "學",
    "kDefinition": "learning, study",
    "kMandarin": "xue2",
    "kJapaneseOn": "ガク",
    "kJapaneseKun": "まな.ぶ",
    "kKorean": "HAK",
    "kSimplifiedVariant": "U+5B66",
    "kRSUnicode": "39.13",
    "kTotalStrokes": "16"
  }
}
```

### 앱용 최소 subset
초기 MVP에선 아래 field만 먼저 뽑아도 충분하다.

- `kDefinition`
- `kMandarin`
- `kJapaneseOn`
- `kJapaneseKun`
- `kKorean`
- `kCantonese`
- `kTraditionalVariant`
- `kSimplifiedVariant`
- `kSemanticVariant`
- `kSpecializedSemanticVariant`
- `kSpoofingVariant`
- `kZVariant`
- `kRSUnicode`
- `kTotalStrokes`
- `kKangXi`
- `kHanYu`
- `kIRG_GSource`, `kIRG_JSource`, `kIRG_KSource`, `kIRG_TSource`

---

## 8. 빠른 요약

- Unihan은 **codepoint 중심 property DB**다.
- 한 줄 형식은 `U+CODEPOINT / kField / value`다.
- 현재 로컬 버전은 Unicode 17.0.0이며, 총 **99개 field**가 확인되었다.
- 가장 중요한 축은:
  - `Readings`
  - `Variants`
  - `IRG Sources / Radical-Stroke`
  - `Dictionary Indices`
  - `Other Mappings`
- Unihan의 최대 강점은 **variant backbone**이다.
- Unihan의 최대 약점은 **정보 completeness의 불균등성**이다.
- 따라서 Sinograph Explorer에서는 Unihan을 **중심 골격**으로 쓰고, e-hanja와 KANJIDIC2로 보강하는 전략이 적합하다.
