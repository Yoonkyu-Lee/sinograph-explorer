"""
KISA SEED block cipher, CBC mode - ported from APK's b5/a.java.
f3202a = 0 → big-endian byte/int conversion.
"""
import struct
from seed_tables import SS0, SS1, SS2, SS3

M32 = 0xFFFFFFFF

# ── Key schedule constants (from o() in b5/a.java) ───────────────────────────
_KC = [
    0x9E3779B9, 0x3C6EF373, 0x78DDE6E6, 0xF1BBCDCD,
    0xE3779B9A, 0xC6EF3734, 0x8DDE6E67, 0x1BBCDC99,  # last = 0x1BBCDC9C - 3?
]
# Exact Java values (converted from signed 32-bit):
_KC_EXACT = [
    (-1640531527) & M32,  # 0x9E3779B9
    1013904243,            # 0x3C6EF373
    2027808486,            # 0x78DDE6E6
    (-239350324) & M32,   # 0xF1BBCDCD
    (-478700647) & M32,   # 0xE3779B99 -- wait, Java says -478700647
    (-957401293) & M32,   # 0xC6EF3733
    (-1914802585) & M32,  # 0x8DDE6E67
    465362127,             # 0x1BBCDC8F
    930724254,             # 0x377998IE
    1861448508,            # 0x6EF3313C
    (-572070280) & M32,   # 0xDDE6637...
    (-1144140559) & M32,  # 0xBBCCCE91
    2006686179,            # 0x779...
    (-281594938) & M32,   # 0xEF333...
    (-563189875) & M32,   # 0xDE667...
]
# The last 2 round keys use constant -1126379749 (Java signed) = 0xBCCCF11B
_KC_LAST  = (-1126379749) & M32  # 0xBCCCF11B
_KC_LAST2 = 1126379749            # positive

# ── G function ────────────────────────────────────────────────────────────────
def _G(x):
    x &= M32
    return (SS0[x & 0xFF] ^ SS1[(x >> 8) & 0xFF] ^ SS2[(x >> 16) & 0xFF] ^ SS3[(x >> 24) & 0xFF]) & M32

# ── Key schedule ─────────────────────────────────────────────────────────────
def _ks_step_i(tmp, rk, idx, key, kc):
    """Equivalent to i() in b5/a.java"""
    kc &= M32
    a = (key[0] + key[2] - kc) & M32
    b = (key[1] + kc - key[3]) & M32
    rk[idx]     = _G(a)
    rk[idx + 1] = _G(b)
    # Shift: rotate key[0:1] right by 8 bits
    k0 = key[0]
    k1 = key[1]
    key[0] = ((k0 >> 8) & 0x00FFFFFF) ^ ((k1 << 24) & M32)
    key[1] = ((k0 << 24) & M32) ^ ((k1 >> 8) & 0x00FFFFFF)

def _ks_step_j(tmp, rk, idx, key, kc):
    """Equivalent to j() in b5/a.java"""
    kc &= M32
    a = (key[0] + key[2] - kc) & M32
    b = (key[1] + kc - key[3]) & M32
    rk[idx]     = _G(a)
    rk[idx + 1] = _G(b)
    # Shift: rotate key[2:3] left by 8 bits
    k2 = key[2]
    k3 = key[3]
    key[2] = ((k2 << 8) & M32) ^ ((k3 >> 24) & 0xFF)
    key[3] = ((k3 << 8) & M32) ^ ((k2 >> 24) & 0xFF)

def _key_schedule(raw_key: bytes) -> list:
    """Compute 32 round keys from 16-byte key. Returns list of 32 unsigned ints."""
    assert len(raw_key) == 16
    # Load key as 4 big-endian 32-bit ints
    key = list(struct.unpack('>4I', raw_key))
    rk = [0] * 32
    tmp = [0, 0]

    _ks_step_i(tmp, rk,  0, key, (-1640531527) & M32)
    _ks_step_j(tmp, rk,  2, key,  1013904243)
    _ks_step_i(tmp, rk,  4, key,  2027808486)
    _ks_step_j(tmp, rk,  6, key, (-239350324) & M32)
    _ks_step_i(tmp, rk,  8, key, (-478700647) & M32)
    _ks_step_j(tmp, rk, 10, key, (-957401293) & M32)
    _ks_step_i(tmp, rk, 12, key, (-1914802585) & M32)
    _ks_step_j(tmp, rk, 14, key, 465362127)
    _ks_step_i(tmp, rk, 16, key, 930724254)
    _ks_step_j(tmp, rk, 18, key, 1861448508)
    _ks_step_i(tmp, rk, 20, key, (-572070280) & M32)
    _ks_step_j(tmp, rk, 22, key, (-1144140559) & M32)
    _ks_step_i(tmp, rk, 24, key, 2006686179)
    _ks_step_j(tmp, rk, 26, key, (-281594938) & M32)
    _ks_step_i(tmp, rk, 28, key, (-563189875) & M32)

    # Last pair (special formula from o())
    kc_last  = (-1126379749) & M32
    kc_last2 = 1126379749
    a = (key[0] + key[2] - kc_last) & M32
    b = (key[1] + kc_last2 - key[3]) & M32
    rk[30] = _G(a)
    rk[31] = _G(b)

    return rk

# ── Round function p() ────────────────────────────────────────────────────────
def _p(state, o0, o1, i0, i1, rk, ki):
    """state[o0] ^= result, state[o1] ^= result2 (from p() in b5/a.java)"""
    A = (state[i0] ^ rk[ki]) & M32
    B = (state[i1] ^ rk[ki + 1]) & M32
    t  = (A ^ B) & M32
    Gt = _G(t)
    u  = (A + Gt) & M32
    Gu = _G(u)
    v  = (Gt + Gu) & M32
    Gv = _G(v)
    w  = (Gu + Gv) & M32
    state[o0] = (state[o0] ^ w) & M32
    state[o1] = (state[o1] ^ Gv) & M32

# ── Block encrypt / decrypt ───────────────────────────────────────────────────
def _block_encrypt(block16: bytes, rk: list) -> bytes:
    """Encrypt one 16-byte block. Equivalent to h() in b5/a.java."""
    s = list(struct.unpack('>4I', block16))
    # 16 rounds with round key pairs at indices 0,2,...,30
    _p(s, 0, 1, 2, 3, rk, 0)
    _p(s, 2, 3, 0, 1, rk, 2)
    _p(s, 0, 1, 2, 3, rk, 4)
    _p(s, 2, 3, 0, 1, rk, 6)
    _p(s, 0, 1, 2, 3, rk, 8)
    _p(s, 2, 3, 0, 1, rk, 10)
    _p(s, 0, 1, 2, 3, rk, 12)
    _p(s, 2, 3, 0, 1, rk, 14)
    _p(s, 0, 1, 2, 3, rk, 16)
    _p(s, 2, 3, 0, 1, rk, 18)
    _p(s, 0, 1, 2, 3, rk, 20)
    _p(s, 2, 3, 0, 1, rk, 22)
    _p(s, 0, 1, 2, 3, rk, 24)
    _p(s, 2, 3, 0, 1, rk, 26)
    _p(s, 0, 1, 2, 3, rk, 28)
    _p(s, 2, 3, 0, 1, rk, 30)
    # Output with swap: [s[2], s[3], s[0], s[1]]
    return struct.pack('>4I', s[2], s[3], s[0], s[1])

def _block_decrypt(block16: bytes, rk: list) -> bytes:
    """Decrypt one 16-byte block. Equivalent to g() in b5/a.java."""
    s = list(struct.unpack('>4I', block16))
    # 16 rounds in reverse: 30, 28, ..., 2, 0
    _p(s, 0, 1, 2, 3, rk, 30)
    _p(s, 2, 3, 0, 1, rk, 28)
    _p(s, 0, 1, 2, 3, rk, 26)
    _p(s, 2, 3, 0, 1, rk, 24)
    _p(s, 0, 1, 2, 3, rk, 22)
    _p(s, 2, 3, 0, 1, rk, 20)
    _p(s, 0, 1, 2, 3, rk, 18)
    _p(s, 2, 3, 0, 1, rk, 16)
    _p(s, 0, 1, 2, 3, rk, 14)
    _p(s, 2, 3, 0, 1, rk, 12)
    _p(s, 0, 1, 2, 3, rk, 10)
    _p(s, 2, 3, 0, 1, rk, 8)
    _p(s, 0, 1, 2, 3, rk, 6)
    _p(s, 2, 3, 0, 1, rk, 4)
    _p(s, 0, 1, 2, 3, rk, 2)
    _p(s, 2, 3, 0, 1, rk, 0)
    # Output with swap
    return struct.pack('>4I', s[2], s[3], s[0], s[1])

# ── XOR 16 bytes ─────────────────────────────────────────────────────────────
def _xor16(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))

# ── CBC encrypt / decrypt ─────────────────────────────────────────────────────
def seed_cbc_encrypt(key: bytes, iv: bytes, plaintext: bytes) -> bytes:
    """Encrypt with PKCS7 padding. key=16 bytes, iv=16 bytes."""
    rk = _key_schedule(key)
    # PKCS7 pad
    pad_len = 16 - (len(plaintext) % 16)
    plaintext += bytes([pad_len] * pad_len)
    ct = b''
    prev = iv
    for i in range(0, len(plaintext), 16):
        block = plaintext[i:i+16]
        block = _xor16(block, prev)
        enc = _block_encrypt(block, rk)
        ct += enc
        prev = enc
    return ct

def seed_cbc_decrypt(key: bytes, iv: bytes, ciphertext: bytes) -> bytes:
    """Decrypt and strip PKCS7 padding."""
    assert len(ciphertext) % 16 == 0
    rk = _key_schedule(key)
    pt = b''
    prev = iv
    for i in range(0, len(ciphertext), 16):
        block = ciphertext[i:i+16]
        dec = _block_decrypt(block, rk)
        pt += _xor16(dec, prev)
        prev = block
    # Strip PKCS7 padding
    pad_len = pt[-1]
    if 1 <= pad_len <= 16:
        pt = pt[:-pad_len]
    return pt


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Test vector from RFC 4269
    # Key:  00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F
    # PT:   00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F
    # CT:   5E BA C6 E0 05 4E 16 68 19 AF F1 CC 6D 34 6C DB
    key_tv = bytes(range(16))
    pt_tv  = bytes(range(16))
    ct_tv  = bytes.fromhex('5EBAC6E0054E166819AFF1CC6D346CDB')

    rk = _key_schedule(key_tv)
    ct = _block_encrypt(pt_tv, rk)
    print(f'Block encrypt test:')
    print(f'  Expected: {ct_tv.hex().upper()}')
    print(f'  Got:      {ct.hex().upper()}')
    print(f'  PASS: {ct == ct_tv}')

    pt2 = _block_decrypt(ct_tv, rk)
    print(f'Block decrypt test:')
    print(f'  Expected: {pt_tv.hex().upper()}')
    print(f'  Got:      {pt2.hex().upper()}')
    print(f'  PASS: {pt2 == pt_tv}')
