# e-hanja / ejajeon CSV Manual

이 문서는 `e-hanja/ejajeon_csv/` 아래에 풀어낸 CSV들이 어떤 구조를 가지는지, 그리고 **특정 한자 1자**를 입력했을 때 관련 정보를 어떤 순서로 찾아야 하는지를 정리한 매뉴얼이다.

기준 원본:
- SQLite DB: [ejajeon_plain.db](./e-hanja/ejajeon_plain.db)
- CSV export dir: [ejajeon_csv](./e-hanja/ejajeon_csv)

핵심 전제:
- CSV에는 SQLite의 물리적 index가 들어 있지 않다.
- 대신 각 CSV 안에는 **논리적 key 컬럼**이 있다.
- 한자 사전 파이프라인의 중심 키는 크게 둘이다.
  - `hanja`: 문자 자체
  - `_id`: 내부 정수 ID

---

## 1. 전체 구조 요약

이 DB는 크게 4개 층으로 볼 수 있다.

1. **한자 마스터 층**
   - `hSchool`
   - 문자 1자에 대한 기본 엔트리

2. **한자 보강 층**
   - `hSchoolCom`, `hCur`, `hTheory`, `hRoot`, `hLaw`, `hLength`, `hDup`
   - 같은 한자에 대한 설명/이체자/이론/음운/중복자 보강

3. **검색/분류 보조 층**
   - `hBusu`, `hBusuAlike`, `hShape`, `hTotal`
   - `srchShp`, `srchSnd`, `srchTotal`
   - `srchWdCr`, `srchWdGr`, `srchWdKey`

4. **단어/전문검색 층**
   - `hWord`, `hWordkey`
   - `ftsHub`, `ftsNatja`, `ftsWord`

추가 자산:
- `imgData_manifest.csv`
- [ejajeon_imgData_png.zip](./e-hanja/ejajeon_imgData_png.zip)
  - stroke 단계별 PNG 이미지 자산

---

## 2. 권장 관계도

```text
                        +------------------+
                        |      hBusu       |
                        | _id              |
                        +---------+--------+
                                  ^
                                  |
                    busu_Id, busu2_Id
                                  |
+-------------+        +----------+-----------+        +----------------+
| hSchoolCom  |<------>|       hSchool        |<------>|     hCur       |
| hanja       | hanja  | _id, hanja           | hanja  | hanja          |
+-------------+        +----------+-----------+        +----------------+
                                  ^
                                  |
                                  | hanja
          +-----------+-----------+-----------+-----------+
          |           |                       |           |
          v           v                       v           v
      hTheory      hRoot                   hLaw       hLength
      hanja        hanja                   hanja      hanja

                                  |
                                  | hanja
                                  v
                            imgData_manifest
                            hanja, stroke

  hSchool._id <----- srchShp.hId     srchShp.shp = shape key

  hWordkey._id <--- srchWdKey.keyId
  hWord._id    <--- srchWdKey.wdId / srchWdCr.wDid / srchWdGr.wdId

  ftsNatja  = search view over ftsHub + hSchool
  ftsWord   = search view over ftsHub + hWord (+ srchWdGr)
```

---

## 3. CSV별 구조 분석

아래에서 `indexing`은 CSV 내부에서 사람이 사용할 때의 **논리적 조회 키**를 뜻한다.

### 3.1 한자 마스터

### `hSchool.csv`
- rows: `10,932`
- fields:
  - `_id`, `hanja`, `hSnd`, `hRead`, `busu_Id`, `hTotal`, `hSame`, `hDiff`, `diffSnd`, `chinaEng`, `english`, `hShape`, `isAni`, `inc`, `busu2_Id`
- role:
  - **한자 1자 기본 엔트리**
  - 실질적인 중심 테이블
- indexing:
  - primary-like key: `_id`
  - natural key: `hanja`
- notes:
  - `busu_Id`, `busu2_Id`는 `hBusu._id`로 연결
  - `hSame`, `hDiff`는 유사/대조 한자를 문자열로 직접 들고 있음

### 3.2 한자 보강

### `hSchoolCom.csv`
- rows: `9,709`
- fields:
  - `hanja`, `hCom`, `hDup`, `hPopular`, `yakja`, `bonja`, `goja`, `waja`, `simpleChina`, `kanji`, `dongja`, `tongja`
- role:
  - 이체자/약자/본자/고자/와자/간체/일본자 등 **비교 정보 보강**
- indexing:
  - key: `hanja`
- join:
  - `hSchoolCom.hanja = hSchool.hanja`
- notes:
  - `hSchool`의 모든 문자에 존재하는 것은 아님

### `hCur.csv`
- rows: `5,978`
- fields:
  - `hanja`, `school`, `school2`, `crHanja`, `crWrite`, `crHanja2`, `crWrite2`
- role:
  - 현용/교육용/급수 분류 보조 데이터
- indexing:
  - key: `hanja`
- join:
  - `hCur.hanja = hSchool.hanja`
- notes:
  - 앱 enum 코드(`jadx_out/sources/n4/a.java`) 기준으로 `crHanja`는 **급수 한자**, `crWrite`는 **급수 쓰기** 코드임이 확인되었다.
  - 공식 코드 라벨:
    - `08=특급`, `09=특급Ⅱ`, `10=1급`, `20=2급`, `30=3급`, `31=3급Ⅱ`,
      `40=4급`, `41=4급Ⅱ`, `50=5급`, `51=5급Ⅱ`, `60=6급`, `61=6급Ⅱ`,
      `70=7급`, `71=7급Ⅱ`, `80=8급`
  - 같은 enum에 `Gov01=중학교용`, `Gov02=고등용`이 존재하므로,
    `school=1`은 **중학교용**, `school=2`는 **고등용**으로 보는 해석이 매우 강하게 지지된다.
  - `school2`, `crHanja2`, `crWrite2`는 이 DB 버전에서는 사실상 미사용이다.

### `hTheory.csv`
- rows: `9,223`
- fields:
  - `hanja`, `hRoot`, `rMade`
- role:
  - 문자 형성 이론/제자 원리 설명
- indexing:
  - key: `hanja`
- join:
  - `hTheory.hanja = hSchool.hanja`
- notes:
  - `hRoot`는 여기서 “형성문자”, “회의문자” 같은 분류 문자열로 쓰이는 경우가 있음
  - `hRoot.csv`의 `hanja`와 직접 FK처럼 연결되는 구조는 아님

### `hRoot.csv`
- rows: `64,665`
- fields:
  - `hanja`, `rOrder`, `rMeaning`, `rSnd`
- role:
  - 한자 1자에 대해 **다중 의미/훈/독음 항목**을 상세하게 풀어놓은 테이블
- indexing:
  - key: `(hanja, rOrder)`
- join:
  - `hRoot.hanja = hSchool.hanja`
- notes:
  - `學`처럼 한 글자에 여러 의미가 있으면 다수 행이 존재
  - 이 DB에서 한국어 자훈/세부 의미를 가장 풍부하게 담고 있는 핵심 테이블 중 하나

### `hLaw.csv`
- rows: `10,031`
- fields:
  - `_id`, `hanja`, `hSnd`, `Origin`, `isRect`
- role:
  - 법원 인명용/정자성/정규화 관련 보조 데이터
- indexing:
  - key candidate: `_id`
  - lookup key: `hanja`
- join:
  - 직접적 FK는 명시적이지 않음
- notes:
  - 앱 query builder(`jadx_out/sources/f4/b.java`)에서 `isRect` 라벨이 직접 확인되었다.
  - 공식 라벨:
    - `0 = 기초`
    - `1 = 추가`
    - `2 = 허용`
  - `isRect=2`일 때 `Origin`은 허용 자형이 연결되는 기준형/원형을 가리키는 경우가 많다.

### `hLength.csv`
- rows: `1,421`
- fields:
  - `hanja`, `sndLength`, `sndExcept`
- role:
  - 음 길이(장단음) 예외 설명
- indexing:
  - key: `hanja`
- join:
  - `hLength.hanja = hSchool.hanja`

### `hDup.csv`
- rows: `477`
- fields:
  - `hMulti`, `checked`, `hanja`
- role:
  - 중복자/동형자 계열 정리
- indexing:
  - composite key-like: `(hMulti, hanja)`
- notes:
  - `checked` 값 예시: `중복자`

### 3.3 부수/형태/기초 분류

### `hBusu.csv`
- rows: `325`
- fields:
  - `_id`, `hanja`, `busu`, `bSub`, `orderBy`, `bMeaning`, `alike`
- role:
  - 부수 마스터
- indexing:
  - key: `_id`
- join:
  - `hSchool.busu_Id -> hBusu._id`
  - `hSchool.busu2_Id -> hBusu._id`

### `hBusuAlike.csv`
- rows: `655`
- fields:
  - `bsGr`, `busu_Id`
- role:
  - 유사 부수 그룹 매핑
- indexing:
  - key-like: `(bsGr, busu_Id)`
- join:
  - `hBusuAlike.busu_Id -> hBusu._id`

### `hShape.csv`
- rows: `305`
- fields:
  - `shp`, `hTotal`
- role:
  - shape key와 총획수 기준 집계/분류 테이블로 보임
- indexing:
  - key: `shp`

### `hTotal.csv`
- rows: `33`
- fields:
  - `total`
- role:
  - 총획수 카탈로그
- indexing:
  - key: `total`

### 3.4 문자 검색 보조

### `srchTotal.csv`
- rows: `2,079`
- fields:
  - `hanja`, `total`, `busu_id`
- role:
  - 총획수/부수 검색 보조
- indexing:
  - key-like: `(hanja, total, busu_id)`
- notes:
  - `hSchool`의 전체 문자 커버리지가 아니라 일부 검색용 subset에 가까움

### `srchShp.csv`
- rows: `86,778`
- fields:
  - `shp`, `hId`
- role:
  - shape key로 한자 `_id`를 찾는 검색용 인덱스
- indexing:
  - key-like: `(shp, hId)`
- join:
  - `srchShp.hId -> hSchool._id`

### `srchSnd.csv`
- rows: `512`
- fields:
  - `sndGr`, `snd`
- role:
  - 독음 그룹/초성 그룹 검색 보조
- indexing:
  - key-like: `(sndGr, snd)`

### 3.5 단어/숙어 사전

### `hWord.csv`
- rows: `85,751`
- fields:
  - `_id`, `word`, `wRead`, `word2`, `wRead2`, `wMeaning`, `book`, `wSame`, `wRelative`, `wEtc`, `gosa`, `wLen`
- role:
  - 단어/숙어/고사성어 사전
- indexing:
  - key: `_id`
- notes:
  - 한자 1자 lookup 테이블이 아니라 어휘 사전 테이블

### `hWordkey.csv`
- rows: `45`
- fields:
  - `_id`, `wKey`, `expl`
- role:
  - 단어 키워드/주제 분류 마스터
- indexing:
  - key: `_id`
- join:
  - 직접 `hWord`와 1:1 또는 FK로 연결되는 구조가 아니다.
  - 실제 연결은 `srchWdKey.keyId -> hWordkey._id` 를 통해 이뤄진다.

### `srchWdCr.csv`
- rows: `85,751`
- fields:
  - `wDid`, `cr`
- role:
  - 단어 검색용 문자 코드 인덱스
- indexing:
  - key-like: `(wDid, cr)`
- join:
  - `srchWdCr.wDid -> hWord._id`

### `srchWdGr.csv`
- rows: `85,751`
- fields:
  - `wdGr`, `wdId`
- role:
  - 단어 그룹 검색 보조
- indexing:
  - key-like: `(wdGr, wdId)`
- join:
  - `srchWdGr.wdId -> hWord._id`

### `srchWdKey.csv`
- rows: `12,362`
- fields:
  - `keyId`, `wdId`
- role:
  - 키워드-단어 연결 인덱스
- indexing:
  - key-like: `(keyId, wdId)`
- join:
  - `srchWdKey.wdId -> hWord._id`

### 3.6 전문검색/파생 검색 뷰

### `ftsHub.csv`
- rows: `105,169`
- fields:
  - `sVal_0`, `sVal_1`, `sVal_2`
- role:
  - SQLite FTS4용 검색 허브
- indexing:
  - CSV에는 없음. 원 DB에서는 `docid`가 중요
- derivation:
  - natja(문자) 검색과 word(단어) 검색의 **공통 검색 corpus**

### `ftsNatja.csv`
- rows: `9,709`
- fields:
  - `um`, `hun`, `jaHae`, `_id`, `hanja`, `hSnd`, `hRead`, `busu_Id`, `hTotal`, `hSame`, `hDiff`, `diffSnd`, `chinaEng`, `english`, `hShape`, `isAni`, `inc`, `busu2_Id`
- role:
  - `ftsHub + hSchool`를 합친 문자 검색용 뷰
- indexing:
  - key: `_id`
  - lookup key: `hanja`, `hun`, `um`
- notes:
  - 실제 앱의 “한자 검색 UI”에 가장 직접적인 검색 테이블

### `ftsWord.csv`
- rows: `95,460`
- fields:
  - `fWord`, `fMeaning`, `wdGr`, `_id`, `word`, `wRead`, `word2`, `wRead2`, `wMeaning`, `book`, `wSame`, `wRelative`, `wEtc`, `gosa`, `wLen`
- role:
  - `ftsHub + hWord (+ srchWdGr)`를 합친 단어 검색용 뷰
- indexing:
  - key: `_id`
  - lookup key: `word`, `fWord`, `fMeaning`

### 3.7 기타

### `imgData_manifest.csv`
- rows: `124,507`
- fields:
  - `hanja`, `codepoint`, `stroke`, `png_size`, `zip_path`
- role:
  - `imgData` BLOB를 PNG로 export한 manifest
- indexing:
  - key-like: `(hanja, stroke)`
- notes:
  - 실제 이미지 파일은 [ejajeon_imgData_png.zip](./e-hanja/ejajeon_imgData_png.zip)에 있음
  - 한 글자당 여러 stroke 단계 PNG가 존재

### `sqlite_stat1.csv`
- rows: `46`
- fields:
  - `tbl`, `idx`, `stat`
- role:
  - SQLite query planner 통계
- indexing:
  - 분석/튜닝용
- notes:
  - 한자 사전 콘텐츠 자체는 아님

---

## 4. `學`을 찾을 때의 정확한 파이프라인

문자 `學`을 조회하는 절차는 아래처럼 정의할 수 있다.

### Step 1. 기준 엔트리 찾기
- 파일: `hSchool.csv`
- 조건: `hanja == '學'`
- 결과:
  - `_id = 2007762`
  - `hSnd = 학`
  - `hRead = 배울 학, 가르칠 교, 고지새 할`
  - `busu_Id = 51`
  - `hTotal = 16`

이 단계에서 `hSchool` 행을 **기준 마스터 레코드**로 삼는다.

### Step 2. 문자 보강 정보 붙이기

#### 2-1. 비교/이체자 보강
- 파일: `hSchoolCom.csv`
- 조건: `hanja == '學'`
- 결과 예:
  - `yakja = 学`
  - `simpleChina = 学`
  - `kanji = 学`
  - `dongja = 敎,斅`

#### 2-2. 현용/서체 보강
- 파일: `hCur.csv`
- 조건: `hanja == '學'`

#### 2-3. 제자 설명
- 파일: `hTheory.csv`
- 조건: `hanja == '學'`
- 결과 예:
  - `hRoot = 회의문자`
  - `rMade = 아이들이 양손에 책을 들고...`

#### 2-4. 뜻/훈의 상세 목록
- 파일: `hRoot.csv`
- 조건: `hanja == '學'`
- 결과:
  - 여러 행
  - `rOrder = 1, 1.1, 1.2, ... 2, 2.1, 3, 3.1`

이 단계가 중요하다. `hSchool.hRead`는 요약 문자열이지만, `hRoot`는 **세부 의미/훈의 정식 목록**이다.

### Step 3. 부수 정보 붙이기
- 파일: `hBusu.csv`
- 조건: `_id == hSchool.busu_Id`
- `學`의 경우:
  - `_id = 51`

필요하면 `busu2_Id`도 같은 방식으로 lookup한다.

### Step 4. shape/검색 관련 연결
- 파일: `srchShp.csv`
- 조건: `shp == '學'`
- 결과:
  - `hId = 2007762`
  - `hId = 2009150`

이 의미:
- shape key `學`에 속하는 한자 집합을 찾는다
- `2009150`은 `hSchool`에서 다시 lookup하면 `斅`

즉 `srchShp`는 직접적인 사전 뜻풀이 테이블이 아니라 **형태 기반 검색 연결**이다.

### Step 5. 단어 사전 확장
문자 `學`가 포함된 단어/숙어를 찾고 싶으면:

- 파일: `hWord.csv`
- 조건:
  - `word` contains `學`
  - or `word2` contains `學`
  - or `wMeaning` contains `學`

이 결과는 수천 건이 나올 수 있다.  
즉 `hWord`는 문자 마스터가 아니라 **어휘 확장용**이다.

### Step 6. 검색 UI용 shortcut
문자 검색만 빠르게 하고 싶다면:

- 파일: `ftsNatja.csv`
- key:
  - `hanja`
  - `hun`
  - `um`

즉 앱 수준에서는 `hSchool + hRoot + hSchoolCom + hTheory + hBusu`를 조합하는 대신,  
검색 엔트리 찾기에는 `ftsNatja.csv`를 먼저 쓰고 나중에 원본 테이블로 내려가는 것도 가능하다.

---

## 5. 권장 조회 규칙

### 한자 1자를 설명하는 최소 파이프라인
1. `hSchool`에서 `hanja` lookup
2. 있으면 `_id`, `busu_Id`, `hRead`, `chinaEng`, `english` 확보
3. `hSchoolCom`, `hCur`, `hTheory`를 `hanja`로 붙임
4. `hRoot`를 `hanja`로 붙여 상세 의미/훈 확장
5. `hBusu`를 `_id = busu_Id`로 붙임
6. 필요하면 `imgData_manifest`로 획순 PNG 연결

### shape/유사자 검색이 필요할 때
1. `srchShp`에서 `shp` lookup
2. 나온 `hId`를 `hSchool._id`로 다시 lookup

### 단어/숙어 예시가 필요할 때
1. `hWord`에서 문자 포함 검색
2. 필요하면 `hWordkey`, `srchWd*`로 분류/검색 보조

### 검색 UI를 재현하고 싶을 때
1. 문자 검색은 `ftsNatja`
2. 단어 검색은 `ftsWord`
3. 원천 정보는 다시 `hSchool`, `hWord`로 내려가서 보여줌

---

## 6. `學` 예시 요약

`學`는 이 구조를 아주 잘 보여준다.

- 기준 행: `hSchool._id = 2007762`
- 이체/약자 정보: `hSchoolCom`
- 현용/코드 보강: `hCur`
- 형성 원리: `hTheory`
- 상세 자훈/뜻 목록: `hRoot`
- 부수 정보: `hBusu._id = 51`
- shape 유사 연결: `srchShp -> 2007762(學), 2009150(斅)`

즉 `學`에 대한 “한 줄 설명”은 `hSchool`에,  
“정교한 사전 정보”는 `hRoot`와 `hSchoolCom`에,  
“검색 보조”는 `srch*`와 `fts*`에 분산되어 있다.

---

## 7. 현재 확실한 점 / 아직 불확실한 점

### 확실한 점
- `hSchool`이 중심 마스터다.
- `hanja`와 `_id` 두 키 체계가 공존한다.
- `hRoot`는 문자별 다중 의미/훈 레이어다.
- `ftsHub/ftsNatja/ftsWord`는 파생 검색 구조다.
- `imgData`는 실제 PNG 자산이다.

### 추가 규명 결과

아래 항목들은 원본 앱 코드를 아직 다 보지 않은 상태에서도, CSV 분포와 조인 결과만으로 꽤 강하게 해석할 수 있는 부분이다.

#### `hShape.csv`
- role:
  - shape search의 **마스터 목록**
- key:
  - `shp`
- interpretation:
  - `shp`는 문자 전체가 아니라 부수/구성요소/필획 모양에 가까운 shape key다.
  - `hTotal`은 그 shape key 자체의 획수로 보인다.
- supporting evidence:
  - 값이 `一`, `丨`, `丶`, `乙`, `⺀`, `亻`, `刂` 같은 shape/radical 계열로 채워져 있음
  - `srchShp.shp`와 직접 연결됨

#### `hTotal.csv`
- role:
  - total-stroke browse의 **기초 마스터**
- key:
  - `total`
- interpretation:
  - `1`부터 `33`까지의 총획수 선택지를 담은 단순 마스터 테이블

#### `srchTotal.csv`
- role:
  - `(total, busu_id)` 기준 browse bucket 인덱스
- logical key:
  - `(total, busu_id)`
- interpretation:
  - 문자 마스터가 아니라, “총획 X + 부수 Y” 조합이 실제로 존재하는지 보여주는 **검색 버킷 테이블**
  - `hanja`는 그 버킷을 표시하기 위한 대표/디스플레이 문자에 가깝다.
- supporting evidence:
  - row 수 `2079`와 `unique(total,busu_id)` 수가 정확히 일치
  - 같은 `hanja`가 여러 `total`에 다시 등장할 수 있음
  - `hSchool` 전체 10,932자 중 일부만 포함하며, browse용 subset 구조와 잘 맞음

#### `srchSnd.csv`
- role:
  - 음 검색(browse-by-reading)의 **마스터 목록**
- logical key:
  - `(sndGr, snd)`
- interpretation:
  - `sndGr`는 초성/첫 음절 기준의 큰 그룹이고, `snd`는 실제 한자음이다.
  - 예: `sndGr=가` 아래에 `가, 각, 간, 갈, 감 ...`이 들어감
- supporting evidence:
  - 512행 전부가 “대표 그룹 -> 실제 음 목록” 구조

#### `hWordkey.csv`
- role:
  - 단어/숙어를 위한 **주제(category) 사전**
- key:
  - `_id`
- interpretation:
  - `_id`는 `hWord._id`와 직접 1:1 대응하는 단어 ID가 아니라, 별도의 keyword/category ID다.
  - `wKey`는 주제명, `expl`은 주제 설명이다.
- examples:
  - `11 = 호칭`
  - `12 = 불교`
  - `17 = 나이`
  - `31 = 학문`
  - `57 = 기타`
- important note:
  - 일부 `_id` 값이 `hWord._id`와 숫자로 겹치기는 하지만, 의미상 FK라고 보면 안 된다.

#### `srchWdKey.csv`
- role:
  - 단어 -> keyword category 매핑
- logical key:
  - `(wdId, keyId)`
- interpretation:
  - `wdId -> hWord._id`
  - `keyId -> hWordkey._id`
- supporting evidence:
  - `srchWdKey.keyId`의 모든 값이 `hWordkey._id`에 정확히 매핑됨
  - `wdId` 하나가 여러 `keyId`를 가질 수 있음

#### `srchWdGr.csv`
- role:
  - 단어 검색을 위한 coarse group 코드
- logical key:
  - `(wdId, wdGr)`
- interpretation:
  - 앱 리소스 기준 `Wordkey = 주제 분류`, `WordkeySub = 한자 성어` 구조가 확인된다.
  - 샘플을 대조하면:
    - `wdGr=0`은 별도 주제 분류가 없는 일반 항목
    - `wdGr=1`은 **주제 분류된 단어** 쪽 항목
    - `wdGr=2`는 **주제 분류된 성어/숙어** 쪽 항목
  - `wdGr=1` 샘플은 `家君`, `家母`, `家父`처럼 `(단어),호칭` 계열이 많고,
    `wdGr=2` 샘플은 `可呵`, `佳境`, `街談`처럼 `(성어),기타/편지/불교` 계열이 많다.
- supporting evidence:
  - `wdGr=0` 행은 거의 전부 `srchWdKey`와 연결되지 않음
  - `wdGr=1,2` 행은 거의 전부 `srchWdKey`와 연결됨
  - `fragment_es_wordkey.xml`와 `i4/l.java`에서 실제 UI가
    **주제 분류 목록 -> 선택된 주제의 하위 한자 성어 목록** 2단으로 구성됨이 확인됨

#### `srchWdCr.csv`
- role:
  - 단어 검색용 code bucket
- logical key:
  - `(wDid, cr)`
- interpretation:
  - 앱 모드 enum(`jadx_out/sources/i4/b.java`)에 `EsIndexCrWd = 급수별 한자성어`가 존재한다.
  - 따라서 `srchWdCr`는 **급수별 한자성어 검색**을 위한 precomputed code bucket으로 보는 해석이 가장 자연스럽다.
  - `cr` 값 체계는 `hCur.crHanja / crWrite`의 급수 코드와 매우 유사하며, 단어를 급수 기준으로 빠르게 필터링하기 위한 인덱스로 보인다.
- measured rule:
  - `srchWdCr` 85,669행을 `hWord.word`의 구성 한자와 `hCur.crHanja`에 대조해보면,
    **97.79%**가 “구성 한자 중 가장 어려운 급수”와 정확히 일치한다.
  - 일치하지 않는 예외는 전부 `cr=00`뿐이며, 이 경우는 모두
    - 구성 한자 중 적어도 하나가 `hCur`에 없어 급수 계산이 불완전하거나
    - 구성 한자 전체가 `hCur` 급수 테이블에 없는 경우였다.
  - 즉 실질적인 규칙은 다음처럼 정리할 수 있다.
    - 각 성어/단어를 구성하는 한자를 본다.
    - `hCur.crHanja`가 존재하는 글자들 중 **가장 어려운 급수**를 대표값으로 쓴다.
    - 글자 중 일부가 급수 테이블에 없으면 `00`으로 빠질 수 있다.
  - 현재까지 decompiled code에서는 `00`에 대응하는 별도 급수 enum/라벨은 발견되지 않았다.
    따라서 `00`은 **미분류/산정불가용 특수값**으로 보는 해석이 가장 자연스럽다.
- confidence:
  - **매우 높음**
  - 공식 검색 모드 이름과 실제 bucket 부여 규칙이 데이터 상으로 거의 확인되었다.

#### `hCur.csv`
- role:
  - 현용/교육용/급수 계열 문자 분류 테이블
- key:
  - `hanja`
- strong hints:
  - `school` 값이 `0,1,2`뿐이며, `1`과 `2`가 각각 정확히 900자
  - 앱 enum에 `Gov01=중학교용`, `Gov02=고등용`이 존재하므로 `school=1 -> 중학교용`, `school=2 -> 고등용` 해석이 매우 강하게 지지된다.
  - `school=0`은 그 외 현용/보조 문자군으로 보는 것이 자연스럽다.
  - `crHanja`, `crWrite`는 각각 공식 급수 라벨을 갖는 코드 체계이며, 이후 `srchWdCr.cr`와도 연결된다.
  - 공식 급수 라벨:
    - `08=특급`, `09=특급Ⅱ`, `10=1급`, `20=2급`, `30=3급`, `31=3급Ⅱ`,
      `40=4급`, `41=4급Ⅱ`, `50=5급`, `51=5급Ⅱ`, `60=6급`, `61=6급Ⅱ`,
      `70=7급`, `71=7급Ⅱ`, `80=8급`
- weaker hints:
  - `crHanja2`, `crWrite2`, `school2`는 이 DB 버전에서는 사실상 미사용

#### `hLaw.csv`
- role:
  - 정자/정규 표기/이체자 관계를 정리한 규범성 보조 테이블로 보임
- key:
  - `_id`
- interpretation:
  - `isRect=2`는 거의 확실하게 **비정규/대체 자형 -> 원형/정자** 링크
  - 이 경우 `Origin`이 대응 기준형을 가리킨다.
  - 앱 query builder 기준 공식 라벨은 `0=기초`, `1=추가`, `2=허용`
- examples:
  - `鑒 -> 鑑`
  - `强 -> 強`
  - `国 -> 國`
  - `教 -> 敎`
- additional hints:
  - `isRect=0(기초)`는 교육용/기초 정자군에 가깝고,
  - `isRect=1(추가)`는 일반 정자군(비교육용 포함 확장 정자군)에 가깝다.
- supporting evidence:
  - `isRect=2` 샘플은 거의 전부 `Origin`이 채워져 있으며, 내용도 정자/원형처럼 보임
  - `isRect=0`은 `hCur.school=1,2`와 많이 겹치고,
  - `isRect=1`은 `school=0` 또는 `hCur` 미등재 문자 비중이 높음

### 아직 불확실한 점
- `srchWdCr.cr=00`을 앱에서 정확히 어떤 라벨/의미로 노출하는지
- `wdGr=1`과 `wdGr=2`의 공식 내부 명칭 문자열이 코드 어딘가에 더 있는지

즉 남은 것은 구조 해석 그 자체라기보다, 몇몇 UI/라벨 수준의 마지막 명칭 확인에 가깝다.
