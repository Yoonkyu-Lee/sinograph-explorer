# Sinograph Explorer Project Plan

이 문서는 현재까지의 조사와 reverse engineering 결과를 바탕으로, **Sinograph Explorer** 프로젝트의 장기 목표와 실제 구현 계획을 정리한 문서다.

이 문서의 목적:
- 우리가 왜 여러 한자 DB를 동시에 분석하고 있는지 잊지 않기
- 각 DB가 어떤 역할을 맡아야 하는지 분명히 하기
- 최종적으로 어떤 앱/데이터/모델을 만들고 싶은지 계속 상기하기

---

## 1. 장기 비전

Sinograph Explorer의 장기 목표는 단순한 “한자 OCR 앱”이 아니다.

우리가 만들고 싶은 것은:
- 희귀자, 이체자, 고자, 지역 특수 한자를 포함한 문자를
- 카메라 또는 이미지에서 인식하고
- 그 문자의
  - 읽기
  - 뜻
  - 자훈
  - 국가별/시대별 표기 차이
  - variant family
  - 사전 인덱스
  - glyph/획순
를 통합적으로 보여주는 **학습형 sinograph explorer**다.

즉 최종 결과물은 다음 두 층을 모두 포함해야 한다.

1. **Dictionary / explorer layer**
   - 한자 자체와 variant family를 설명하는 지식 시스템
2. **Vision / OCR layer**
   - 실제 화면/문헌/사진에서 그 문자를 읽어오는 인식 시스템

---

## 2. 핵심 문제의식

현재 존재하는 어떤 단일 DB도 우리 목적을 완전히 만족시키지 못한다.

### Unihan의 한계
- codepoint backbone과 variant 정보는 강하지만,
- 정보가 sparse하고,
- 한국어 정보와 human-friendly 설명이 빈약하다.

### e-hanja의 강점과 한계
- 한국어 자훈/뜻풀이/변이 정보가 풍부하다.
- 하지만 앱용 관계형 DB라 구조 해석이 필요했고,
- 공개 표준 메타데이터처럼 다루기엔 재구성이 필요하다.

### KANJIDIC2의 강점과 한계
- 일본 온훈독, 병음, dictionary reference가 잘 정리되어 있다.
- 하지만 일본 중심 편향이 강하고,
- variant 의미가 Unihan처럼 직접적이지 않다.

따라서 프로젝트의 정답은
- “하나의 완벽한 DB를 찾는 것”
이 아니라,
- **여러 DB의 교집합과 보완 관계를 이용해 통합 sinograph DB를 만드는 것**
이다.

---

## 3. 현재 확보한 핵심 데이터 소스

### 3.1 Unihan
- 역할:
  - Unicode backbone
  - variant graph backbone
  - IRG source / radical / stroke / dictionary index 보조
- 문서:
  - [UNIHAN_MANUAL.md](../db_src/UNIHAN_MANUAL.md)

### 3.2 e-hanja / ejajeon
- 역할:
  - 한국어 자훈
  - 한국어 뜻풀이
  - 교육용/급수/정자/허용자형 정보
  - 단어/성어/획순 이미지 자산
- 문서:
  - [EHANJA_MANUAL.md](../db_src/EHANJA_MANUAL.md)

### 3.3 KANJIDIC2
- 역할:
  - 일본 온독/훈독
  - 병음, 한국 독음, 베트남 한자음 보강
  - dictionary reference와 glyph search code 보강
- 문서:
  - [KANJIDIC2_MANUAL.md](../db_src/KANJIDIC2_MANUAL.md)

---

## 4. 통합 DB의 목표 구조

최종적으로는 각 문자에 대해 다음과 같은 canonical record를 만들고 싶다.

```text
CharacterRecord
  character
  codepoint
  sources_present

  readings
    mandarin
    cantonese
    korean_hangul
    korean_romanized
    japanese_on
    japanese_kun
    vietnamese

  meanings
    english
    korean_explanation
    korean_hun

  variant_family
    traditional_variants
    simplified_variants
    semantic_variants
    z_variants
    spoofing_variants
    representative_form

  structural_info
    radical
    total_strokes
    alternate_strokes
    shape_codes

  classification
    school_grade
    exam_grade
    jouyou / jinmeiyo / education_hanja flags

  references
    kangxi
    hanyu
    morohashi
    nelson
    halpern
    ...

  media
    stroke_images
    glyph_variants
```

즉 목표는 단순 merge가 아니라,
**서로 다른 사전 체계를 하나의 공통 record model로 정규화하는 것**이다.

---

## 5. DB 통합 전략

### 5.1 1차 key는 Unicode codepoint
가능한 한 모든 문자의 1차 식별자는 Unicode codepoint로 맞춘다.

이유:
- Unihan이 codepoint 중심
- KANJIDIC2도 `literal + ucs`를 제공
- e-hanja도 결국 문자 자체를 기반으로 역매핑 가능

즉 canonical key는:

```text
U+5B78
U+6588
...
```

### 5.2 variant family는 Unihan 우선
variant 관계의 의미는 Unihan이 제일 직접적이므로:
- `kTraditionalVariant`
- `kSimplifiedVariant`
- `kSemanticVariant`
- `kSpecializedSemanticVariant`
- `kZVariant`
- `kSpoofingVariant`

를 variant graph backbone으로 삼는다.

### 5.3 한국어 의미/자훈은 e-hanja 우선
한국어 쪽은 현재까지:
- Unihan보다 e-hanja가 훨씬 풍부하고
- user-facing 앱에 더 적합한 표현을 준다.

따라서:
- 훈
- 한국어 설명
- 교육/급수 분류
- 허용/기초/추가 자형
은 e-hanja를 우선한다.

### 5.4 일본/다국어 reading 및 dictionary refs는 KANJIDIC2 보강
다음 정보는 KANJIDIC2를 보강 소스로 쓴다.
- `ja_on`
- `ja_kun`
- `pinyin`
- `korean_h`, `korean_r`
- `vietnam`
- Nelson / Halpern / Morohashi / SKIP / Four-Corner

---

## 6. Sinograph Explorer 앱 목표

최종 앱은 사용자가 문자를 넣으면 최소한 다음을 보여줘야 한다.

1. **인식된 문자**
2. **기본 정보**
   - codepoint
   - 부수
   - 획수
3. **읽기**
   - 중국어
   - 한국어
   - 일본어
4. **뜻과 자훈**
5. **variant family**
   - 직접 variant
   - 연결된 family 전체
   - representative/reference form
6. **사전 reference**
7. **획순/자형 이미지**

즉 앱은 단순 DB browser가 아니라,
**variant-aware educational character explorer**가 되어야 한다.

---

## 7. OCR / 비전 시스템과의 연결

DB 통합은 최종 목표가 아니라 OCR 시스템을 위한 기반이기도 하다.

### 이유 1. 인식 후 explanation
OCR이 rare character를 읽었다고 끝나는 것이 아니라,
그 문자를 사용자가 이해할 수 있게 설명해야 한다.

즉:
- OCR output -> codepoint
- codepoint -> integrated sinograph DB
- DB -> human-friendly explanation

### 이유 2. 학습 데이터 생성
희귀자 OCR 모델을 훈련하려면:
- 어떤 문자를 class로 잡을지
- 어떤 variant family를 묶을지
- 어떤 글자들이 실제로 uncommon long tail인지
를 DB 통합 결과를 바탕으로 판단해야 한다.

### 이유 3. synthetic glyph generation
훈련용 synthetic dataset을 만들 때도:
- uncommon 한자의 목록
- representative form과 variant form
- codepoint 기반 렌더링 가능 여부
를 통합 DB에서 가져와야 한다.

즉 통합 DB는 단순 사전이 아니라
**rare-character OCR pipeline의 label space definition layer**이기도 하다.

---

## 8. 단계별 실행 계획

### Phase 1. DB 구조 완전 정리
- Unihan 구조 매뉴얼
- e-hanja 구조 매뉴얼
- KANJIDIC2 구조 매뉴얼

현재 상태:
- Unihan: 정리 중/완료 예정
- e-hanja: 완료
- KANJIDIC2: 완료

### Phase 2. 공통 스키마 설계
- 각 DB 필드를 공통 schema에 매핑
- 어떤 source가 어떤 field의 authority인지 결정

예:
- variant = Unihan authority
- Korean explanation = e-hanja authority
- ja_on/ja_kun = KANJIDIC2 authority

### Phase 3. 통합 parser / merger 구현
- source별 parser 작성
- canonical codepoint 기준 merge
- conflict resolution 규칙 설계

### Phase 4. Explorer 앱 데이터 백엔드 구축
- lookup API / local DB
- variant graph
- direct info + best linked info fallback

### Phase 5. OCR 학습용 문자 집합 선정
- integrated DB에서 uncommon character set 정의
- variant family와 representative forms 정리
- synthetic rendering class list 구축

### Phase 6. OCR 실험과 연결
- rare/uncommon classes synthetic render
- fallback classifier 훈련
- Sinograph Explorer DB와 결과 연결

---

## 9. 현재 우리가 잊지 말아야 할 것

이 프로젝트의 진짜 목표는:
- “OCR 정확도 조금 올리기”
만이 아니다.

우리가 하려는 것은:
- **한자를 문자 체계로 이해할 수 있는 데이터 기반 explorer를 만들고**
- 그걸 바탕으로
- **희귀/변체 한자 인식 시스템까지 연결하는 것**이다.

즉 DB 정리 작업은 부수 작업이 아니라,
프로젝트 전체의 기반 공사다.

---

## 10. 빠른 요약

- 단일 DB로는 부족하다.
- Unihan, e-hanja, KANJIDIC2를 결합해야 한다.
- canonical key는 가능한 한 Unicode codepoint로 맞춘다.
- Unihan은 variant backbone, e-hanja는 한국어 의미 backbone, KANJIDIC2는 reading/reference backbone이다.
- 최종 목표는 integrated sinograph DB + explorer + rare-character OCR pipeline이다.
