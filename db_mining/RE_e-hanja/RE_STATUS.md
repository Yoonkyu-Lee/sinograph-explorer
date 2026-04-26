# e-hanja 앱 리버스 엔지니어링 현황 보고서

작성일: 2026-04-02  
최종 업데이트: 2026-04-03 10:00

> 이 폴더(`lab3/db_mining/RE_e-hanja`)는 e-hanja 리버스 엔지니어링과 원본 추출 작업물을 보관하는 작업장이다.
> 여기서 구조를 파악하고 검증한 뒤, 재사용 가능한 DB와 매뉴얼만 `lab3/db_src`로 승격한다.

---

## 최종 목적

**e-hanja 디지털 한자사전 앱(`e-hanja사전_1.25.14.apk`)의 자전 DB(`ejajeon.dat`)를 추출하여 로컬에서 직접 쿼리 가능한 형태로 확보**

이 DB는 앱 설치 후 서버에서 다운로드되며, SQLCipher로 암호화되어 있다.
앱 없이 DB를 직접 활용하는 것이 목표다.

---

## 완전히 해결된 것 ✅

### 1. APK 서명 기반 AES 키 도출

- `Signature.toByteArray()` → SHA-256 → 32-byte AES/ECB 키
- AES 키: `8c373aa43d3625bd1b6b6105376d77971e1a683038d40061916f6373ccfeb6df`

### 2. 암호화 리소스 복호화 완료

| 리소스 | 용도 | 복호화 결과 |
|--------|------|------------|
| `schoolkuk` | SEED 키 | `708cc0690e84f83e578e2e827dfa1dc8` |
| `schoolkiv` | SEED IV | `34abb122029c2c34e00046daa11b5c67` |
| `schoolr` | SQLCipher passphrase 원본 | `34abb122029c2c34e00046daa11b5c67` |
| `schoolgpk` | RSA 공개키 (과금) | `0V7qMkxGtOWIRcHEb4tleTCe4wIDAQAB` |
| `schoolgslt` | 과금용 salt (SQLCipher 무관) | `6B7D96232AF8FA2510561BB3BA168BD2A632EDAF` |

`a5.a.a(ctx, "schoolr")` 반환값 = `"x'34abb122029c2c34e00046daa11b5c67"` (닫는 `'` 없음)

### 3. KISA SEED/CBC Python 구현

- `seed_tables.py`: b5/a.java에서 추출한 S-박스 4개 (SS0~SS3)
- `seed_cipher.py`: 키 스케줄, 라운드 함수(p), CBC encrypt/decrypt
- round-trip 검증 통과

### 4. 서버 통신 프로토콜 해독

- **요청**: `PREFIX(16B) + JSON` → SEED/CBC encrypt → hex → POST body
- **응답**: hex → SEED/CBC decrypt → JSON
- **GET 다운로드**: `.download.Rest.asp` 엔드포인트에 GET 직접 접근 시 파일 스트리밍 발생

### 5. ehanja.user.dat SQLCipher 해독 성공 ✅

### 6. ejajeon.dat (자전DB) 추출 및 해독 성공 ✅ (2026-04-03)

**획득 방법**: 안드로이드 에뮬레이터(Android 28 default) + APK 패치(라이선스 체크 bypass)  
→ 앱이 서버에서 자동 다운로드한 정상 파일을 `adb pull`로 추출

**파일 정보:**
```
파일명 : ejajeon_app.dat
크기   : 238,456,832 bytes (← 서버 info API와 정확히 일치)
위치   : /sdcard/Android/data/kr.openmindgna.ehanjab/files/ejajeon.dat
```

**SQLCipher 열기 성공 파라미터 (ehanja.user.dat와 동일):**
```
passphrase : x'34abb122029c2c34e00046daa11b5c67   ← 닫는 quote 없음
page_size  : 4096
kdf_iter   : 256000
algorithm  : PBKDF2_HMAC_SHA512
```

**스키마 개요 (69개 항목 확인):**
```
테이블: hTotal(33), hWord(85,751), hBusu(325), hSchool(10,932),
        hShape, hRoot, hLength, hLaw, hCur, hDup, hBusuAlike,
        hTheory, hSchoolCom, hWordkey, imgData, ftsHub(FTS), srch* 인덱스 등
뷰    : ftsNatja, ftsWord
```

**이전 347MB 파일이 실패했던 이유**: 서버가 파일을 직접 스트리밍할 때 정상적인 ejajeon.dat가 아닌 다른 파일을 반환했던 것으로 추정. 앱을 통해 정상 경로로 받은 파일(238MB)은 동일 파라미터로 정상 열림.

**파라미터:**
```
passphrase : x'34abb122029c2c34e00046daa11b5c67   ← 닫는 quote 없음
page_size  : 4096
kdf_iter   : 256000
algorithm  : PBKDF2_HMAC_SHA512
```

**Python 열기 코드:**
```python
import sqlcipher3
conn = sqlcipher3.connect('ehanja.user.dat')
c = conn.cursor()
c.execute("PRAGMA key = \"x'34abb122029c2c34e00046daa11b5c67\";")
c.execute("PRAGMA cipher_page_size = 4096;")
c.execute("PRAGMA kdf_iter = 256000;")
c.execute("PRAGMA cipher_kdf_algorithm = PBKDF2_HMAC_SHA512;")
```

**실행 결과 (2026-04-03 재현):**
```
ehanja.user.dat: OPEN SUCCESS, sqlite_master rows = 6
```

**스키마:**
```sql
CREATE TABLE android_metadata (locale TEXT)
CREATE TABLE bookmark (tbl TEXT, mHanja TEXT, checked INTEGER DEFAULT 0, timestamp DATETIME DEFAULT 0)
CREATE TABLE noGlyph (hanja TEXT)
CREATE UNIQUE INDEX idx_bookmark_01 on bookmark (tbl, mHanja)
CREATE INDEX idx_bookmark_02 on bookmark (timestamp desc)
CREATE UNIQUE INDEX idx_noGlyph_01 on noGlyph (hanja)
```

---

## 서버 Info 엔드포인트 응답 (2026-04-03 재현)

**요청:**
```
GET http://m.openmindGnA.kr/ehanja/gna.ehanja.app.info.Rest.asp
    ?package=kr.openmindgna.ehanja&version=1.25.14
User-Agent: Dalvik/2.1.0 (Linux; U; Android 13; SM-G991B Build/TP1A.220624.014)+openmindgna;+ehanja
```

**응답 (JSON):**
```json
{
  "info": {
    "app": {
      "version": "1.25.14",
      "published_date": "2026-01-26 12:00:00"
    },
    "db": {
      "published_date": "2025-12-01 12:00:00",
      "file_size": "238456832"
    }
  }
}
```

→ **서버가 공식으로 알려준 ejajeon.dat 크기: `238,456,832` bytes (약 227MB)**

---

## ejajeon.dat 현황 및 의심 근거

### 파일을 어떻게 받았나

[download_ejajeon_get.sh](download_ejajeon_get.sh) 참고.  
요약: `.download.Rest.asp` 엔드포인트에 curl로 GET 요청. 소요 시간 약 43분 (2026-04-02 21:20 ~ 22:04).

```bash
curl -L -o ejajeon.dat \
  -H "User-Agent: Dalvik/2.1.0 (Linux; U; Android 13; SM-G991B Build/TP1A.220624.014)+openmindgna;+ehanja" \
  "http://m.openmindGnA.kr/ehanja/gna.ehanja.app.download.Rest.asp"
```

**추가 확인 (2026-04-03):** POST로 보내도 서버는 동일하게 347MB 바이너리를 그대로 스트리밍한다 (요청 본문 무시). 즉 이 엔드포인트는 메타데이터 API가 아니라 그냥 파일 서버다. GET과 POST 모두 동일한 파일을 반환한다.

```
POST .download.Rest.asp → HTTP 200, 347,125,760 bytes
응답 첫 64바이트: c9424ac8de74f301 7c9a80df6d4a8cc2 ...  ← ejajeon.dat 헤더와 동일
```

→ **sUrl 같은 건 없다. 이 파일이 서버에서 받을 수 있는 ejajeon.dat 전부다.**  
→ **남은 문제는 순수하게 SQLCipher 키/파라미터 하나다.**

### 파일 분석 결과 (2026-04-03 재현)

```
Size          : 347,125,760 bytes (331.04 MB)
MD5           : 079b01ee2fa45dfaeffd4a6fe0d87617
Header[0:32]  : c9424ac8de74f3017c9a80df6d4a8cc2051862ee59c45c38a31efb786d6322b2
Header[0:16]  : c9424ac8de74f3017c9a80df6d4a8cc2  (← SQLCipher salt 위치)
Entropy(100KB): 7.9984 bits/byte  (← 거의 완전 암호화)
SQLite header : NOT FOUND (파일 전체 347MB 탐색)
size % 1024   : 0
size % 4096   : 2048
size / 1024   : 338,990 pages
```

### SQLCipher 열기 시도 결과 (2026-04-03 재현)

동일 키·파라미터로 user.dat는 성공, ejajeon.dat는 실패:

```
ehanja.user.dat : OPEN SUCCESS  (동일 passphrase, page=4096, SHA512/256000)
ejajeon.dat     : OPEN FAIL - file is not a database  (152가지 파라미터 조합 전부 실패)
```

### 의심 근거 요약

| 항목 | 서버 공식 값 | 실제 파일 |
|------|-------------|----------|
| 파일 크기 | 238,456,832 bytes | 347,125,760 bytes |
| 크기 차이 | — | **+109MB 초과** |
| SQLite 헤더 | 있어야 함 (암호화되어 있어도 위치는 고정) | **없음** |
| SQLCipher 열기 | 가능해야 함 (동일 코드 경로) | **152가지 전부 실패** |

→ GET으로 받은 347MB 파일은 정상적인 ejajeon.dat가 아닐 가능성이 높다.  
→ **단, 파일명 `ejajeon.dat`은 임의로 지은 것이 아니라 `EhanjaProvider.java` 코드에 하드코딩된 이름**(`l5.a.d(applicationContext) + "ejajeon.dat"`)이므로, 이것이 메인 자전 DB의 실제 파일명임은 확실하다.

---

## 🎉 최종 목적 달성 (2026-04-03)

**ejajeon.dat (자전DB) 추출 + SQLCipher 해독 완료.** 85,751개 단어, 325개 부수, 10,932개 교과 한자 데이터 로컬에서 직접 쿼리 가능.

```python
# 로컬 열기 코드 (ejajeon_app.dat 기준)
import sqlcipher3
conn = sqlcipher3.connect('ejajeon_app.dat')
c = conn.cursor()
c.execute("PRAGMA cipher_page_size = 4096;")
c.execute("PRAGMA kdf_iter = 256000;")
c.execute("PRAGMA cipher_kdf_algorithm = PBKDF2_HMAC_SHA512;")
c.execute("PRAGMA key = \"x'34abb122029c2c34e00046daa11b5c67\";")
```

---

## 이전에 막혀있던 부분 (해결됨) ✅

### SEED POST 통신 — 해결됨 (단, 결과가 예상과 다름)

**2026-04-03 확인:** POST 통신 자체는 성공한다. 문제는 서버가 SEED 암호화 JSON을 반환하는 것이 아니라, **요청 본문(SEED 암호화 여부)을 무시하고 무조건 347MB 바이너리를 스트리밍**한다는 것이다.

```
POST .download.Rest.asp (SEED 암호화 요청 전송)
→ HTTP 200, 347,125,760 bytes  ← JSON이 아니라 파일 그대로
→ 첫 바이트: c9424ac8... (ejajeon.dat 헤더와 동일)
```

→ `sUrl`이라는 개념 자체가 없었다. 이 엔드포인트 = 파일 서버.  
→ **우리가 가진 347MB 파일이 이 서버에서 얻을 수 있는 ejajeon.dat의 전부다.**

### ejajeon.dat SQLCipher 열기 실패 (미해결 🔴)

남은 문제는 순수하게 SQLCipher 파라미터 하나다.

- `ehanja.user.dat` (28KB, 사용자 DB): SHA512/256000/page=4096 → **성공**
- `ejajeon.dat` (347MB, 메인 자전 DB): 동일 파라미터 포함 152가지 조합 → **전부 실패**

두 파일이 동일 코드 경로(`u4.b.C0120b`)로 열리는데 ejajeon.dat만 실패하는 이유가 불명확하다.

**새로 발견된 결정적 단서들 (2026-04-03):**

#### 1. 앱이 기대하는 ejajeon.dat 크기 하드코딩 (`o4/a.java`)

```java
f6862c = new m5.c(1, l5.a.d(context) + "ejajeon.dat", 238456832);
// m5.c(type=1, path, expected_size=238456832)
```

`m5.c.a()` 유효성 검사:
```java
if (file.isFile() && file.length() == 238456832) → 유효
else → 손상/미완료, 재다운로드 시도
```

→ 앱 기준으로 우리 파일(347MB)은 "손상된 파일". 앱이 기대하는 ejajeon.dat = **정확히 238,456,832 bytes**.

#### 2. 347MB 파일은 page_size=4096이 불가능

```
347,125,760 % 4096 = 2048  → page_size=4096 불가
347,125,760 % 1024 = 0     → page_size=1024 가능
347,125,760 % 2048 = 0     → page_size=2048 가능
```

ehanja.user.dat(열기 성공)는 page=4096이었는데 ejajeon.dat는 그것이 불가능하다.  
→ **두 파일이 다른 SQLCipher 파라미터로 암호화되어 있을 가능성이 높다.**

#### 3. 238MB 경계 분석

238,456,832 = 1024의 배수 = 4096의 배수. 경계 전후가 모두 고엔트로피 바이너리.  
나머지 조각(347 - 238 = 108MB)도 독립된 파일처럼 보이나 어느 쪽도 SQLCipher로 열리지 않음.

#### 4. passphrase 검증

- 모든 코드 경로에서 `a5.a.a(ctx, "schoolr")` = `x'34abb122029c2c34e00046daa11b5c67` (34B) 사용 — 예외 없음
- raw hex key(`x'....'` closing quote 있음), raw 16B, ascii hex 등 다양한 형태 시도 → 전부 실패
- **passphrase 자체는 맞는 것으로 판단. 파일 또는 파라미터 문제.**

#### 5. 수동 PBKDF2+AES 복호화 실패

ehanja.user.dat(열기 성공 기준)에서도 수동 복호화가 맞지 않음.  
→ SQLCipher 내부 KDF/IV 방식이 단순 PBKDF2+CBC와 다름. libsqlcipher.so 커스텀 빌드(OpenSSL 3.5.4) 영향 가능성.

#### 종합 판단

| 가설 | 근거 |
|------|------|
| **서버가 현재 다른 버전 파일을 서빙** | info API 파일크기(238MB)와 실제 수신 파일(347MB) 불일치. 앱 업데이트 후 DB도 커졌을 가능성 |
| **ejajeon.dat는 page=1024, 구버전 SQLCipher** | 크기 배수 제약 상 page=4096 불가. 앱 빌드 당시 구버전으로 암호화한 채 서버에 올려둔 것 |
| **347MB = 구DB(238MB) + 신DB(108MB) 이어붙임** | 경계가 정확히 1024/4096 배수, 두 번째 조각도 독립 파일처럼 시작 |

---

## 서버 구조

| 엔드포인트 | 방식 | 역할 |
|-----------|------|------|
| `.info.Rest.asp` | GET (쿼리스트링) | DB 버전/크기 조회 → **성공 확인** |
| `.download.Rest.asp` | POST 또는 GET | 요청 방식 무관하게 347MB 파일 스트리밍 → **파일 서버 역할** |
| `fonts/gna.app.font.Download.asp` | GET | 폰트 다운로드 |

**User-Agent 필수:** `Dalvik/2.1.0 (...)+openmindgna;+ehanja`

---

## DEX 주요 클래스

```
a5/a.java  — APK 서명 기반 AES/ECB 복호화
a5/b.java  — SEED 키/IV 관리
b5/a.java  — KISA SEED 핵심 (S-박스, 라운드, CBC)
n5/a.java  — hex 문자열 ↔ bytes
z4/a.java  — HTTP 통신 (SEED 암호화 POST)
r4/c.java  — Application.onCreate() 진입점
u4/b.java  — SQLiteOpenHelper wrapper (ejajeon.dat 열기)
EhanjaProvider.java — ContentProvider (ejajeon.dat + userdb ATTACH)
```

---

## DB 스키마 (DEX SQL 문자열에서 파악)

```sql
-- ejajeon.dat 예상 테이블
hW, hS              -- 한자 메인/서브
hschool             -- 학교용 한자
hSnd, soundBase     -- 발음
busuBase            -- 부수
strokeBase          -- 획수
shapeBase, diffshapeBase
popularBase, samemeaningBase
hanjungilBase, lawBase

-- ehanja.user.dat (ATTACH as userdb) ← 열기 성공
bookmark            -- 사용자 북마크
noGlyph             -- 글리프 없는 한자
```

---

## 전략 실행 결과

**전략 A 성공** — 에뮬레이터에서 앱 실행 + APK 패치로 라이선스 우회 → 정상 파일 획득

---

## 이전 전략 메모 (참고용, 이미 해결됨)

### 전략 A: 앱 실제 실행 (가장 확실)
안드로이드 기기(또는 에뮬레이터)에서 앱을 실제 설치·실행하여 ejajeon.dat를 다운받은 뒤:
1. `adb pull` 로 파일 추출 → 크기/MD5 확인
2. `adb shell`에서 `PRAGMA cipher_version;` 등으로 실제 파라미터 확인
- **장점**: 가장 확실하고 추측 불필요
- **단점**: 기기 필요, 루팅 또는 adb backup 권한 필요할 수 있음

### 전략 B: 구버전 APK 다운 후 비교
앱 버전이 업데이트되면서 DB도 달라졌다면, 구버전 APK에서 expected_size=238456832와 일치하는 빌드를 찾아 서버에서 238MB짜리 파일을 받기.
- APKPure, APKMirror 등에서 구버전 APK 탐색
- 구버전 APK의 `schoolr` 리소스가 다를 경우 passphrase도 다를 수 있음

### 전략 C: DB Browser for SQLCipher 로 직접 시도
Python sqlcipher3이 이 파일과 호환 안 될 가능성. GUI 도구로 다양한 설정을 직접 입력해보기.
- DB Browser for SQLCipher 설치 후 ejajeon_part1.dat(238MB 조각) 열기 시도
- 설정: Encryption Key = `x'34abb122029c2c34e00046daa11b5c67`, page size = 1024, cipher = SQLCipher3

### 전략 D: libsqlcipher.so 동적 분석
앱이 실제로 sqlite3_key_v2()를 호출할 때 전달하는 바이트를 Frida 등으로 후킹하여 확인.
- 에뮬레이터에 앱 설치 → Frida로 `sqlite3_key_v2` 후킹 → 실제 key bytes 캡처
- **장점**: passphrase 추측 불필요, 100% 확실
- **단점**: Frida 환경 세팅 필요

---

## 작업 파일 (리버스 엔지니어링 작업시 추가되는 것들은 계속 업데이트 할것)

```
lab3/
├── e-hanja사전_1.25.14.apk
├── RE_STATUS.md                   ← 이 파일 (분석 결과 요약)
├── RE_WORKFLOW.md                 ← 리버스 엔지니어링 워크플로우 상세 기술 문서
├── jadx-1.5.5/
├── jadx_out/                      ← 디컴파일 결과
│   ├── sources/                   ← Java 소스
│   └── resources/res/values/strings.xml  ← 암호화 리소스
├── seed_tables.py                 ← SEED S-박스 (b5/a.java 추출)
├── seed_cipher.py                 ← Python SEED/CBC 구현
├── decrypt_and_download.py        ← 서버 SEED POST 통신 스크립트
├── download_ejajeon_get.sh        ← GET 직접 다운로드 스크립트 (재현용)
├── try_open_db.py                 ← SQLCipher 파라미터 탐색 (일반)
├── try_ejajeon.py                 ← ejajeon.dat 전용 파라미터 탐색 (152가지)
├── ejajeon.dat                    ← GET으로 받은 파일 (347MB, 정상 파일 아님)
├── ejajeon_app.dat                ← 에뮬레이터에서 앱이 다운로드한 정상 파일 ✅ (238MB, SQLCipher 열기 성공)
├── ejajeon_plain.db               ← 암호화 없는 일반 SQLite export ✅ (235MB, sqlcipher3 불필요)
├── ehanja.user.dat                ← APK 내 동봉 (28KB, SQLCipher 열기 성공 ✅)
├── libsqlcipher.so                ← APK에서 추출한 ARM64 네이티브 라이브러리
├── apktool.jar                    ← apktool 2.9.3
├── ehanja_patched_final.apk       ← 라이선스 체크 bypass 패치된 APK (서명됨)
└── .venv/                         ← Python venv (pycryptodome, cryptography, sqlcipher3)
```

doc, small_benchmark_test, Unihan, unihan_graph_viewer는 현재 e-hanja 사전 리버스 엔지니어링 작업과는 무관하므로 무시.
