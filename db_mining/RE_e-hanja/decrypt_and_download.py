#!/usr/bin/env python3
"""
e-hanja 앱 다운로드 엔드포인트 복호화 + DB URL 추출 스크립트
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import zipfile, hashlib, base64, json, time, urllib.request
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
from seed_cipher import seed_cbc_encrypt, seed_cbc_decrypt

# ── 1. APK 서명으로 AES 복호화 → SEED 키/IV ──────────────────────────────────
APK_PATH = "e-hanja사전_1.25.14.apk"

ROOT = Path(__file__).resolve().parent
APK_PATH = ROOT / APK_PATH

with zipfile.ZipFile(APK_PATH) as z:
    cert_rsa = z.read('META-INF/CERT.RSA')

from cryptography.hazmat.primitives.serialization import pkcs7
from cryptography.hazmat.primitives.serialization import Encoding
certs = pkcs7.load_der_pkcs7_certificates(cert_rsa)
cert_der = certs[0].public_bytes(Encoding.DER)
aes_key = hashlib.sha256(cert_der).digest()

SCHOOLKUK = '3UXjjmTTTo7+EP+ExY/zRESCdNZ8X43HtLUq2mGVonQtiuGqcu89c2fdlts0uCWY'
SCHOOLKIV = 'oSNrRSIvMuGsdGC6nmLF3yjueDOYNOTtl0+bsjDL0LJhjfBnFL3ktV1pEjSTGX6h'

def aes_ecb_decrypt(b64_ct):
    enc = base64.b64decode(b64_ct)
    return unpad(AES.new(aes_key, AES.MODE_ECB).decrypt(enc), 16).decode('utf-8')

def parse_hex_csv(s):
    return bytes(int(x, 16) for x in s.split(','))

SEED_KEY = parse_hex_csv(aes_ecb_decrypt(SCHOOLKUK))
SEED_IV  = parse_hex_csv(aes_ecb_decrypt(SCHOOLKIV))
print(f"SEED KEY: {SEED_KEY.hex()}")
print(f"SEED IV:  {SEED_IV.hex()}")

# ── 2. 요청/응답 암호화 헬퍼 ─────────────────────────────────────────────────
PREFIX = b'0123456789abcdef'

def encrypt_request(payload: dict) -> bytes:
    """JSON → prefix 추가 → SEED/CBC 암호화 → hex 인코딩"""
    pt = PREFIX + json.dumps(payload, separators=(',', ':')).encode('utf-8')
    ct = seed_cbc_encrypt(SEED_KEY, SEED_IV, pt)
    return ct.hex().encode('ascii')

def decrypt_response(resp_bytes: bytes) -> dict:
    """hex 응답 → hex 디코딩 → SEED/CBC 복호화 → JSON"""
    hex_str = resp_bytes.decode('utf-8').strip()
    ct = bytes.fromhex(hex_str)
    pt = seed_cbc_decrypt(SEED_KEY, SEED_IV, ct)
    # prefix 16바이트 제거
    json_bytes = pt[len(PREFIX):].rstrip(b'\x00').strip()
    return json.loads(json_bytes.decode('utf-8'))

# ── 3. 서버 POST 요청 ────────────────────────────────────────────────────────
UA   = 'Dalvik/2.1.0 (Linux; U; Android 13; SM-G991B Build/TP1A.220624.014)+openmindgna;+ehanja'
BASE = 'http://m.openmindGnA.kr/ehanja/gna.ehanja.app'
PKG  = 'kr.openmindgna.ehanja'

def post_encrypted(suffix, payload):
    url  = BASE + suffix
    body = encrypt_request(payload)
    print(f"\nPOST {url}")
    print(f"  Body (first 64 chars): {body[:64].decode()}...")
    req = urllib.request.Request(url, data=body,
          headers={'User-Agent': UA, 'Content-Type': 'application/octet-stream'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    print(f"  Response size: {len(raw)} bytes")
    print(f"  Response (first 64 chars): {raw[:64]}")
    return raw

# ── 4. req-nonce ─────────────────────────────────────────────────────────────
raw1 = post_encrypted('.download.Rest.asp', {
    "method": "req-nonce", "package": PKG, "reqNonce": 0, "reqTmStmp": 0,
})

try:
    resp1 = decrypt_response(raw1)
    print(f"\n[req-nonce 복호화 성공]")
    print(json.dumps(resp1, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"[req-nonce 복호화 실패] {e}")
    print(f"  Raw hex: {raw1[:128].hex()}")
    # 응답이 plain text인지 확인
    try:
        print(f"  Raw text: {raw1[:200].decode('utf-8', errors='replace')}")
    except:
        pass
    sys.exit(1)

# ── 5. req-download ──────────────────────────────────────────────────────────
res_nonce = resp1.get('resNonce', 0)
res_ts    = resp1.get('resTmStmp', 0)

raw2 = post_encrypted('.download.Rest.asp', {
    "method": "req-download", "package": PKG,
    "reqNonce": res_nonce, "reqTmStmp": res_ts,
})

try:
    resp2 = decrypt_response(raw2)
    print(f"\n[req-download 복호화 성공]")
    print(json.dumps(resp2, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"[req-download 복호화 실패] {e}")
    print(f"  Raw hex: {raw2[:128].hex()}")
    try:
        print(f"  Raw text: {raw2[:300].decode('utf-8', errors='replace')}")
    except:
        pass
    sys.exit(1)

# ── 6. 다운로드 URL 추출 ─────────────────────────────────────────────────────
down_files = resp2.get('downFiles', [])
if not down_files:
    print(f"\n[오류] downFiles 없음")
    print(json.dumps(resp2, indent=2, ensure_ascii=False))
else:
    print(f"\n{'='*50}")
    print(f"다운로드 파일 목록 ({len(down_files)}개):")
    for i, f in enumerate(down_files):
        print(f"\n  [{i+1}]")
        print(f"    URL     : {f.get('sUrl', '?')}")
        print(f"    크기    : {f.get('length', '?'):,} bytes")
        print(f"    저장경로: {f.get('destPath', '?')}")
        print(f"    제품    : {f.get('sProduct', '?')}")
        print(f"    메서드  : {f.get('srvMethod', '?')}")
    print(f"{'='*50}")
