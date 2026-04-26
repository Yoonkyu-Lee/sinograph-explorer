# Android 앱 리버스 엔지니어링 워크플로우

**대상**: e-hanja 디지털 한자사전 앱 (`e-hanja사전_1.25.14.apk`)  
**목표**: 앱 내 SQLCipher 암호화 DB 추출 및 로컬 쿼리  
**작성일**: 2026-04-03  
**환경**: Windows 11, WSL/Git Bash, Android SDK

> 이 문서의 명령은 기본적으로 `lab3/db_mining/RE_e-hanja`에서 실행한다.
> `db_mining`은 리버스 엔지니어링/추출 작업물 보관소이고, 여기서 검증된 결과만 `lab3/db_src`로 옮겨 정리한다.

---

## 전체 흐름 개요

```
APK 입수
  ↓
정적 분석 (jadx) — 소스코드 복원, 암호화 키·프로토콜 역추적
  ↓
동적 분석 준비 — Android 에뮬레이터 세팅
  ↓
APK 패치 (apktool) — 라이선스 체크 bypass
  ↓
패치 APK 서명·설치 → 앱 실행 → DB 자동 다운로드
  ↓
adb pull — 파일 추출
  ↓
SQLCipher 파라미터 적용 → 복호화·쿼리
  ↓
일반 SQLite export
```

---

## Phase 1: 정적 분석 (Static Analysis)

### 1-1. jadx로 APK 디컴파일

**jadx**는 APK의 DEX 바이트코드를 Java 소스로 복원한다.

```bash
# jadx 실행 (GUI)
./jadx-1.5.5/bin/jadx-gui e-hanja사전_1.25.14.apk

# 또는 CLI로 sources/ 폴더에 출력
./jadx-1.5.5/bin/jadx -d jadx_out e-hanja사전_1.25.14.apk
```

**결과물**: `jadx_out/sources/` — 패키지 구조 그대로 Java 코드 복원됨

**주요 탐색 포인트:**
- `AndroidManifest.xml` — 패키지명, 진입 Activity, 권한
- `res/values/strings.xml` — 하드코딩된 문자열 (암호화된 리소스 포함)
- `Application.onCreate()` 진입점 찾기 → 초기화 흐름 추적

### 1-2. 암호화된 리소스 역추적

앱이 SQLCipher 비밀번호를 평문으로 두지 않고 APK 서명의 SHA-256을 AES 키로 써서 암호화한 hex 문자열을 리소스에 저장함.

```
strings.xml에서 발견된 암호화 리소스:
  schoolkuk  = "7aa1680d..." (SEED 키)
  schoolkiv  = "cb3e6a8e..." (SEED IV)
  schoolr    = "ab12cd34..." (SQLCipher passphrase)
  schoolgpk  = "4d6f7e9a..." (Google Play 공개키)
```

**복호화 로직 (`a5/a.java` 분석):**

```java
// 1. APK 서명 → SHA-256 → 32바이트 AES 키
byte[] sig = context.getPackageManager()
    .getPackageInfo(packageName, GET_SIGNATURES)
    .signatures[0].toByteArray();
MessageDigest md = MessageDigest.getInstance("SHA-256");
byte[] aesKey = md.digest(sig);

// 2. AES/ECB 복호화
Cipher cipher = Cipher.getInstance("AES/ECB/PKCS5Padding");
cipher.init(DECRYPT_MODE, new SecretKeySpec(aesKey, "AES"));
byte[] plain = cipher.doFinal(Hex.decode(encryptedHex));

// 3. 반환: "x'" + hex(plain)  ← 닫는 quote 없음 (SQLCipher 특수 형식)
return "x'" + Hex.toHexString(plain);
```

**Python으로 재현:**

```python
import hashlib
from Crypto.Cipher import AES

# APK에서 추출한 서명 바이트 (apksigner 또는 keytool로 획득)
sig_bytes = bytes.fromhex("...")  # APK 서명 DER 인코딩
aes_key = hashlib.sha256(sig_bytes).digest()

cipher = AES.new(aes_key, AES.MODE_ECB)
plain = cipher.decrypt(bytes.fromhex(encrypted_hex))
# PKCS5 패딩 제거
plain = plain[:-plain[-1]]
passphrase = "x'" + plain.hex()
# → "x'34abb122029c2c34e00046daa11b5c67"
```

### 1-3. SQLCipher 파라미터 파악

**`u4/b.java`** (SQLiteOpenHelper 래퍼) 분석:

```java
// SQLCipher 기본값 사용 확인
new C0120b(context, path, null, version)
// null hook → 라이브러리 기본값 적용
```

**`net/zetetic/database/sqlcipher/SQLiteDatabase.java`** 분석:
- `getBytes(str)` → UTF-8 변환 후 `sqlite3_key_v2()` 호출
- PRAGMA 별도 설정 없음 → **SQLCipher 4.x 기본값 사용**

**SQLCipher 4.x 기본 파라미터:**
```
cipher_page_size = 4096
kdf_iter         = 256000
kdf_algorithm    = PBKDF2_HMAC_SHA512
hmac_algorithm   = HMAC_SHA512
```

### 1-4. 서버 통신 프로토콜 분석

**`z4/a.java`** — HTTP 통신 레이어:

```
요청: JSON → SEED/CBC 암호화 → hex 인코딩 → POST body
응답: hex → SEED/CBC 복호화 → JSON
```

**SEED 키/IV 도출** (`b5/a.java`, `a5/b.java`):
```
SEED 키 = AES/ECB 복호화(schoolkuk)
SEED IV  = AES/ECB 복호화(schoolkiv)
```

**`b5/a.java`**에서 SEED S-박스 4개(SS0~SS3) 직접 추출 → `seed_tables.py`로 저장.

---

## Phase 2: 동적 분석 환경 구축

### 2-1. Android 에뮬레이터 세팅

**에뮬레이터 선택 기준:**
- `google_apis_playstore` 이미지: Play Store 지원, `adb root` 불가 → **부적합**
- `default` 이미지: Play Store 없음, `adb root` 가능 → **적합**

```bash
# 1. 시스템 이미지 다운로드
$ANDROID_SDK/cmdline-tools/latest/bin/sdkmanager \
  "system-images;android-28;default;x86_64"

# 2. AVD 생성
$ANDROID_SDK/cmdline-tools/latest/bin/avdmanager create avd \
  --name ehanja_test \
  --package "system-images;android-28;default;x86_64" \
  --device "pixel_2"

# 3. 에뮬레이터 실행
$ANDROID_SDK/emulator/emulator -avd ehanja_test -no-snapshot-load &

# 4. 부팅 대기
adb wait-for-device
adb shell getprop sys.boot_completed  # "1"이 될 때까지 대기
```

### 2-2. APK 설치 및 기본 확인

```bash
# 설치
adb install e-hanja사전_1.25.14.apk

# 패키지명 확인 (코드에 명시된 kr.openmindgna.ehanja와 다름!)
adb shell pm list packages | grep hanja
# → package:kr.openmindgna.ehanjab  ← 'b'가 붙어있음

# 앱 실행
adb shell am start -n \
  "kr.openmindgna.ehanjab/kr.openmindgna.ehanja.startup.SplashActivity"

# 실시간 로그 모니터링
adb logcat | grep -i "ehanja\|openmind\|licens"
```

---

## Phase 3: APK 패치 (License Check Bypass)

### 3-1. 문제 파악

앱 실행 시 Google Play 라이선스 체크("라이선스 확인" 다이얼로그) 발생.  
에뮬레이터는 Play Store 없음 → 체크 항상 실패 → 앱 진입 불가.

**라이선스 체크 흐름 (`startup/a.java` 분석):**
```
SplashActivity.I0()
  → y4.b.e()        // LVL 체크 시작 (Google Play 서버 통신)
    → SplashActivity$a.a(String)  // 콜백: 라이선스 OK → K0() 호출
    → SplashActivity$a.d(boolean, String)  // 콜백: 라이선스 없음 → 에러
```

### 3-2. apktool로 smali 추출

**apktool**은 APK를 smali 바이트코드(Dalvik 어셈블리)로 디컴파일한다.

```bash
# 다운로드
curl -L -o apktool.jar \
  https://github.com/iBotPeaches/Apktool/releases/download/v2.9.3/apktool_2.9.3.jar

# APK → smali (--no-res: 리소스 디코딩 생략, 속도↑)
java -jar apktool.jar d e-hanja사전_1.25.14.apk -o apktool_out --no-res -f
```

**결과물**: `apktool_out/smali/` — 패키지별 `.smali` 파일

### 3-3. smali 패치

**smali**는 Dalvik VM의 어셈블리 언어다. Java 1줄 ≈ smali 수 줄.

**패치 1: `I0()` — LVL 체크 제거**

`apktool_out/smali/kr/openmindgna/ehanja/startup/SplashActivity.smali`:

```smali
# 원본: I0()가 y4.b.e()를 호출해서 Google Play LVL 체크 시작
.method public I0()V
    .locals 8
    # ... 8개 레지스터로 LVL 체크 파라미터 준비 후 y4.b.e() 호출
    invoke-static/range {v1 .. v7}, Ly4/b;->e(...)V
    return-void
.end method

# 패치: LVL 체크 건너뛰고 DB 상태 확인 후 바로 K0() 호출
.method public I0()V
    .locals 2

    # m5.a.b(context).a() → DB 상태 정수값
    invoke-static {p0}, Lm5/a;->b(Landroid/content/Context;)Lm5/a;
    move-result-object v0
    invoke-virtual {v0}, Lm5/a;->a()I
    move-result v1
    invoke-virtual {p0, v1}, Lkr/openmindgna/ehanja/startup/SplashActivity;->K0(I)V

    return-void
.end method
```

**패치 2: `SplashActivity$a.d()` — "미인가" 콜백 리다이렉트 (보조)**

`apktool_out/smali/kr/openmindgna/ehanja/startup/SplashActivity$a.smali`:

```smali
# 원본: d(boolean, String)가 "라이선스 없음" 메시지 발송
.method public d(ZLjava/lang/String;)V
    .locals 1
    # ... Handler에 0x2c 메시지 → 앱 종료 흐름
.end method

# 패치: "라이선스 OK" 경로인 a()를 그대로 호출
.method public d(ZLjava/lang/String;)V
    .locals 0
    invoke-virtual {p0, p2}, Lkr/openmindgna/ehanja/startup/SplashActivity$a;->a(Ljava/lang/String;)V
    return-void
.end method
```

**smali 문법 핵심:**
```
p0 = this (instance method의 첫 번째 인자)
p1, p2 ... = 메서드 파라미터
v0, v1 ... = 로컬 레지스터 (.locals N 으로 선언)
invoke-virtual {인스턴스, 파라미터...}, 클래스;->메서드명(시그니처)반환타입
move-result-object vN = 직전 invoke 반환값(object) 저장
```

### 3-4. 재빌드 및 서명

```bash
# 1. smali → DEX 재컴파일
java -jar apktool.jar b apktool_out -o ehanja_patched_unsigned.apk

# 2. 패치된 classes.dex를 원본 APK에 이식
#    (apktool 재빌드 시 리소스 누락 문제 회피용)
python3 << 'EOF'
import zipfile

with open('apktool_out/build/apk/classes.dex', 'rb') as f:
    patched_dex = f.read()

with zipfile.ZipFile('e-hanja사전_1.25.14.apk', 'r') as orig:
    with zipfile.ZipFile('ehanja_patched2.apk', 'w',
                         compression=zipfile.ZIP_DEFLATED) as out:
        for entry in orig.infolist():
            if entry.filename == 'classes.dex':
                # DEX는 ZIP_STORED (정렬 요건)
                out.writestr(zipfile.ZipInfo('classes.dex'),
                             patched_dex, compress_type=zipfile.ZIP_STORED)
            else:
                out.writestr(entry, orig.read(entry.filename))
EOF

# 3. 테스트 키 생성
keytool -genkeypair -keystore test.keystore -alias testkey \
  -keyalg RSA -keysize 2048 -validity 10000 \
  -storepass testpass -keypass testpass \
  -dname "CN=Test, O=Test, C=KR"

# 4. APK 서명 (apksigner)
$ANDROID_SDK/build-tools/36.1.0/apksigner.bat sign \
  --ks test.keystore --ks-key-alias testkey \
  --ks-pass pass:testpass --key-pass pass:testpass \
  --out ehanja_patched_final.apk ehanja_patched2.apk

# 5. 설치 (기존 앱 제거 후)
adb uninstall kr.openmindgna.ehanjab
adb install ehanja_patched_final.apk
```

**왜 원본 APK에 dex만 교체하나?**

Windows 파일시스템은 대소문자 구분 없음. 원본 APK에는 `res/sr.xml`과 `res/Sr.xml`이 별도 파일로 존재하는데, apktool이 extract 시 둘 다 같은 경로로 취급해 하나가 덮임. 재빌드 시 누락 → 앱 크래시. 리소스에는 손대지 않고 DEX만 교체하면 이 문제를 완전히 회피할 수 있다.

---

## Phase 4: 데이터 추출

### 4-1. 앱 실행 및 DB 다운로드 유도

```bash
# 앱 실행
adb shell am start -n \
  "kr.openmindgna.ehanjab/kr.openmindgna.ehanja.startup.SplashActivity"

# 에뮬레이터에서 "자전DB 설치" 팝업 → "지금 설치" 탭
adb shell input tap 505 769   # 화면 좌표로 버튼 탭 (해상도에 따라 조정)

# 다운로드 진행 모니터링
watch -n 5 'adb shell "ls -la \
  /sdcard/Android/data/kr.openmindgna.ehanjab/files/ejajeon.dat"'
```

### 4-2. 스크린샷으로 상태 확인

```bash
# 에뮬레이터 스크린샷 캡처
adb exec-out screencap -p > screen.png
```

### 4-3. adb pull로 파일 추출

```bash
# 앱 외부 저장소 (/sdcard) → 로컬
# 주의: Git Bash(Windows)에서 /sdcard/... 경로 앞에 // 필요
adb pull //sdcard/Android/data/kr.openmindgna.ehanjab/files/ejajeon.dat \
  ejajeon_app.dat

# 앱 내부 저장소 접근 (default 이미지에서 adb root 가능)
adb root
adb pull /data/data/kr.openmindgna.ehanjab/files/somefile.db localfile.db
```

**adb 유용한 명령어 모음:**

```bash
# 연결된 기기 목록
adb devices

# 실시간 로그 (태그 필터)
adb logcat -s "MyApp:D"

# 전체 로그 (특정 PID)
adb logcat --pid=$(adb shell pidof kr.openmindgna.ehanjab)

# 파일 목록 (앱 외부 저장소)
adb shell ls -la /sdcard/Android/data/kr.openmindgna.ehanjab/files/

# 파일 목록 (앱 내부 저장소, root 필요)
adb shell ls -la /data/data/kr.openmindgna.ehanjab/

# APK 정보
adb shell dumpsys package kr.openmindgna.ehanjab | grep -E "version|path"

# 앱 강제 종료
adb shell am force-stop kr.openmindgna.ehanjab

# 화면 크기 확인 (tap 좌표 계산용)
adb shell wm size
```

---

## Phase 5: SQLCipher 복호화

### 5-1. 파라미터 확인 및 열기

```python
# pip install sqlcipher3-binary  (Python 3.10~3.12)
import sqlcipher3

conn = sqlcipher3.connect('ejajeon_app.dat')
c = conn.cursor()

# 순서 중요: key PRAGMA 이후에 나머지 설정
c.execute("PRAGMA cipher_page_size = 4096;")
c.execute("PRAGMA kdf_iter = 256000;")
c.execute("PRAGMA cipher_kdf_algorithm = PBKDF2_HMAC_SHA512;")
c.execute("PRAGMA key = \"x'34abb122029c2c34e00046daa11b5c67\";")
# ↑ 닫는 ' 없음 — SQLCipher raw hex key 형식 (앱 코드의 반환값 그대로)

c.execute("SELECT count(*) FROM sqlite_master;")
print(c.fetchone())  # (69,) ← 성공
```

**passphrase 형식 설명:**
- `"password"` — 일반 텍스트 비밀번호 (PBKDF2 파생)
- `"x'HEXHEX'"` — raw hex key (KDF 없이 직접 사용)
- `"x'HEXHEX"` ← 이 앱은 닫는 `'`가 없음. `getBytes()` 호출 시 `x'...` 자체를 UTF-8로 변환해서 `sqlite3_key_v2()`에 전달하기 때문에 SQLCipher가 그냥 문자열로 처리함.

### 5-2. 파라미터를 모를 때 — 탐색 방법

```python
# SQLCipher 버전별 기본값
compat_modes = {
    1: {"page": 1024, "iter": 4000,   "alg": "PBKDF2_HMAC_SHA1"},
    2: {"page": 1024, "iter": 4000,   "alg": "PBKDF2_HMAC_SHA1"},
    3: {"page": 1024, "iter": 64000,  "alg": "PBKDF2_HMAC_SHA1"},
    4: {"page": 4096, "iter": 256000, "alg": "PBKDF2_HMAC_SHA512"},
}

for ver, cfg in compat_modes.items():
    try:
        conn = sqlcipher3.connect(db_path)
        c = conn.cursor()
        c.execute(f"PRAGMA cipher_page_size = {cfg['page']};")
        c.execute(f"PRAGMA kdf_iter = {cfg['iter']};")
        c.execute(f"PRAGMA cipher_kdf_algorithm = {cfg['alg']};")
        c.execute(f'PRAGMA key = "{passphrase}";')
        c.execute("SELECT count(*) FROM sqlite_master;")
        print(f"Version {ver}: SUCCESS")
        break
    except:
        print(f"Version {ver}: FAIL")
    finally:
        conn.close()
```

### 5-3. 암호화 없는 SQLite로 export

```python
import sqlcipher3, sqlite3

src = sqlcipher3.connect('ejajeon_app.dat')
sc = src.cursor()
sc.execute("PRAGMA cipher_page_size = 4096;")
sc.execute("PRAGMA kdf_iter = 256000;")
sc.execute("PRAGMA cipher_kdf_algorithm = PBKDF2_HMAC_SHA512;")
sc.execute("PRAGMA key = \"x'34abb122029c2c34e00046daa11b5c67\";")

dst = sqlite3.connect('ejajeon_plain.db')
dc = dst.cursor()

# 스키마 복사
sc.execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL "
           "AND name != 'sqlite_stat1' ORDER BY type DESC, name;")
for (sql,) in sc.fetchall():
    try: dc.execute(sql)
    except: pass

# 데이터 복사
sc.execute("SELECT name FROM sqlite_master WHERE type='table' "
           "AND name != 'sqlite_stat1';")
for (tbl,) in sc.fetchall():
    sc.execute(f"SELECT * FROM {tbl};")
    rows = sc.fetchall()
    if rows:
        ph = ",".join(["?"] * len(rows[0]))
        dc.executemany(f"INSERT OR REPLACE INTO {tbl} VALUES ({ph})", rows)
    dst.commit()

dst.execute("ANALYZE;")
dst.commit()
# 이후 일반 sqlite3으로 바로 사용 가능
```

---

## 핵심 도구 정리

| 도구 | 용도 | 획득 |
|------|------|------|
| **jadx** | APK → Java 소스 복원 (정적 분석) | github.com/skylot/jadx |
| **apktool** | APK → smali 바이트코드 (패치용) | github.com/iBotPeaches/Apktool |
| **apksigner** | APK 서명 | Android SDK build-tools 포함 |
| **keytool** | 서명 키스토어 생성 | JDK 포함 |
| **adb** | 에뮬레이터/기기 통신 | Android SDK platform-tools 포함 |
| **avdmanager** | 에뮬레이터 AVD 생성 | Android SDK cmdline-tools 포함 |
| **sqlcipher3** | SQLCipher DB 열기 (Python) | pip install sqlcipher3-binary |

---

## 배운 개념 요약

### Android APK 구조
```
APK (= ZIP 파일)
├── classes.dex         ← Dalvik 바이트코드 (앱 로직 전체)
├── AndroidManifest.xml ← 패키지명, 컴포넌트, 권한 선언
├── res/                ← 리소스 (레이아웃, 문자열, 이미지)
├── assets/             ← 번들 파일
└── lib/                ← 네이티브 .so 라이브러리
```

### Smali = Dalvik 어셈블리
- Java 코드는 `.class` → `dex2jar` → `.dex` 로 컴파일됨
- Smali는 `.dex`의 사람이 읽을 수 있는 표현
- apktool이 `.dex` ↔ `.smali` 변환을 담당
- 코드 패치 = smali 텍스트 편집 → 재컴파일

### Google Play LVL (License Verification Library)
- 앱이 구매됐는지 Google Play 서버에 확인하는 메커니즘
- `LicenseChecker.checkAccess()` → Google 서버 → `allow()` / `dontAllow()` 콜백
- 에뮬레이터(Play 서비스 없음) → 항상 `dontAllow()` → 앱 차단
- 우회: `dontAllow()` 콜백 코드를 `allow()` 경로로 리다이렉트

### SQLCipher
- SQLite의 AES-256 암호화 확장
- 파일 첫 16바이트 = KDF salt (일반 SQLite의 "SQLite format 3\000" 헤더 없음)
- `PRAGMA key`로 비밀번호/키 설정, 나머지 PRAGMA로 KDF 파라미터 지정
- 버전 4.x 기본: PBKDF2_SHA512, 256000 iterations, page=4096

### adb (Android Debug Bridge)
- USB 또는 TCP로 Android 기기/에뮬레이터와 통신
- `adb shell` — 원격 셸 실행
- `adb pull/push` — 파일 전송
- `adb install/uninstall` — APK 관리
- `adb logcat` — 실시간 로그
- `adb root` — root 권한 (debug/emulator 빌드에서만 가능)

---

## 이 워크플로우가 적용되는 다른 케이스

- 앱 내 SQLite/SQLCipher DB에 오프라인으로 접근하고 싶을 때
- 앱의 네트워크 통신 프로토콜(암호화 포함)을 분석할 때
- 루팅 없이 앱 데이터를 에뮬레이터로 추출할 때
- 유료 앱 기능을 연구/교육 목적으로 분석할 때
- CTF (Capture The Flag) Android 문제 풀이
