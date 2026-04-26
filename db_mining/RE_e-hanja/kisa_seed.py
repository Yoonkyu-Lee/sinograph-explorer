"""
KISA SEED (RFC 4269) pure-Python implementation.
Standard 128-bit block cipher, 128-bit key, 16 rounds, Feistel.
"""

import struct

# ── S-boxes (standard KISA SEED SS0, SS1) ────────────────────────────────────
SS0 = [
    0x2989A1A8,0x05858184,0x16C6D2D4,0x13C3D3D0,0x14445054,0x1D0D111C,0x2C8CA0AC,0x25052124,
    0x1D4D515C,0x03434340,0x18081018,0x1E0E121C,0x11415150,0x3CCCF0FC,0x0ACAC2C8,0x23436360,
    0x28082028,0x04444044,0x20002020,0x1D8D919C,0x20C0E0E0,0x22C2E2E0,0x08C8C0C8,0x17071314,
    0x2585A1A4,0x0F8F838C,0x03030300,0x3B4B7378,0x3B8BB3B8,0x13031310,0x12C2D2D0,0x2ECEE2EC,
    0x30407070,0x0C8C808C,0x3F0F333C,0x28880808, # ... shortened for illustration
    # Full table continues...
]

# Instead of hardcoding, let's use the computed tables from the spec.
# We'll implement the G function from 2 base S-boxes.

# SEED uses two 8-to-32-bit S-boxes (KC matrices) derived from GF(2^8) operations.
# The 4-table optimization: precompute rotations so:
#   G(x) = SS0[x[0]] ^ SS1[x[1]] ^ SS0[x[2]] ^ SS1[x[3]]  (where x[0] is MSB byte)
# But we implement using the pre-computed tables from the reference implementation.

# Reference tables from KISA SEED specification / RFC 4269
_SS0 = [
    0x2989A1A8, 0x05858184, 0x16C6D2D4, 0x13C3D3D0, 0x14445054, 0x1D0D111C, 0x2C8CA0AC, 0x25052124,
    0x1D4D515C, 0x03434340, 0x18081018, 0x1E0E121C, 0x11415150, 0x3CCCF0FC, 0x0ACAC2C8, 0x23436360,
    0x28082028, 0x04444044, 0x20002020, 0x1D8D919C, 0x20C0E0E0, 0x22C2E2E0, 0x08C8C0C8, 0x17071314,
    0x2585A1A4, 0x0F8F838C, 0x03030300, 0x3B4B7378, 0x3B8BB3B8, 0x13031310, 0x12C2D2D0, 0x2ECEE2EC,
    0x30407070, 0x0C8C808C, 0x3F0F333C, 0x28880808, 0x1E8E929C, 0x28C8E0E8, 0x14C4D0D4, 0x32C2F2F0,
    0x1C4C505C, 0x0404040C, 0x36363630, 0x0F0F030C, 0x1C0C101C, 0x3A4A7278, 0x0E8E828C, 0x0D0D010C,
    0x24C4E0E4, 0x23C3E3E0, 0x10405050, 0x1B4B5158, 0x21C1E1E0, 0x08080008, 0x2C4C606C, 0x1D5D515C,
    0x30003030, 0x3484B0B4, 0x0B0B030C, 0x29C9A1A8, 0x02828280, 0x14040410, 0x21414160, 0x22C2E2E0, # dup?
    # This approach is getting complex. Let me use the actual tables from the Java code.
]

# The Java code's f3202a=0 means big-endian int representation.
# c(x)=x&0xFF (byte0=LSB), d(x)=(x>>8)&0xFF, e(x)=(x>>16)&0xFF, f(x)=(x>>24)&0xFF (byte3=MSB)
# G(x) = f3203b[x&0xFF] ^ f3204c[(x>>8)&0xFF] ^ f3205d[(x>>16)&0xFF] ^ f3206e[(x>>24)&0xFF]
# The 4 tables are 4 different "columns" of the combined S-box matrix.

# Rather than hardcode all 4 tables here (they're huge), use the 2-S-box + rotation approach.
# KISA defines:
#   G(x) applied to 4 bytes: G(a,b,c,d) = z where z uses SS0 and SS1
# Actually the 4-table version is equivalent to:
#   T0[b0] = SS0[b0] rotated 0
#   T1[b1] = SS1[b1] rotated 8 (or different)
# etc.

# Let me use the two base 8->32 substitution tables from RFC 4269 and compute G directly.

# From RFC 4269 / KISA specification:
# The G function uses two substitution boxes KC[0][256] and KC[1][256],
# each mapping 8-bit input to 32-bit output.

# For SEED, these are the standard tables. Let me embed them.
# Below are KC[0] and KC[1] from the official KISA SEED reference code.
_KC0 = [
    0x9E3779B9, 0x6B901122, 0x00000000, 0x13198A2E, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    # ... incomplete
]

# This approach is getting too complex with manually typing 512 table entries.
# Let me use a different strategy: derive the tables algorithmically from GF(2^8).

# KISA SEED S-boxes are defined as follows in the spec:
# S1 and S2 are 4-bit to 4-bit substitutions.
# SS0[x] = S1[x_hi] * 256^3 + S2[x_lo] * 256^2 + S1[x_hi xor S2[x_lo]] * 256 + S2[S1[x_hi] xor x_lo]
# where x_hi = x >> 4, x_lo = x & 0xF

# The 4-bit S-boxes S1 and S2:
_S1_4bit = [0xA, 0xD, 0xE, 0x6, 0xF, 0x4, 0x9, 0x1, 0x5, 0x2, 0x0, 0x3, 0x7, 0xC, 0x8, 0xB]
_S2_4bit = [0xE, 0xC, 0xB, 0x0, 0x9, 0x2, 0xD, 0xB, 0x3, 0x1, 0x5, 0xF, 0x8, 0x6, 0x4, 0x7] # wrong?

# Actually, I'll take a completely different and correct approach:
# Port the exact Java implementation including all 4 large S-box arrays.
# The Java source has them — I just need to extract them.
# But they're embedded in the class file. Let me use a script to extract from the jadx output.

raise NotImplementedError(
    "Use seed_impl.py which has the actual table values extracted from b5/a.java"
)
