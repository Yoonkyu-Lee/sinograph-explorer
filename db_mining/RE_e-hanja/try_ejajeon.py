#!/usr/bin/env python3
"""
Try to open ejajeon.dat with correct PRAGMA ordering.
SQLCipher requires: set cipher params FIRST, then PRAGMA key.

Known facts:
  - ehanja.user.dat opens with: page=4096, SHA512, 256000 iters, passphrase (no closing quote)
  - ejajeon.dat: 347,125,760 bytes = multiple of 1024 but NOT of 4096 → page_size=1024
  - Both use passphrase: x'34abb122029c2c34e00046daa11b5c67 (no closing quote)
  - Both opened via same SQLiteOpenHelper with no custom hook (library defaults)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import sqlcipher3

PASS_NO_QUOTE = "x'34abb122029c2c34e00046daa11b5c67"  # no closing quote (what app passes)
PASS_WITH_QUOTE = "x'34abb122029c2c34e00046daa11b5c67'"  # with closing quote (raw hex flag)
RAW_HEX = "34abb122029c2c34e00046daa11b5c67"

ROOT = Path(__file__).resolve().parent
DB = ROOT / "ejajeon.dat"

def try_open(label, setup_fn):
    try:
        conn = sqlcipher3.connect(str(DB))
        c = conn.cursor()
        setup_fn(c)
        c.execute("SELECT count(*) FROM sqlite_master;")
        row = c.fetchone()
        conn.close()
        print(f"[SUCCESS] {label} → count={row[0]}")
        return True
    except Exception as e:
        err = str(e).split('\n')[0][:80]
        print(f"[FAIL] {label}: {err}")
        return False

# Correct order: cipher params FIRST, then key
configs = []

# V4 defaults but with page=1024 (key last)
for ps in [1024, 2048, 4096, 512, 256]:
    for kdf in ['SHA512', 'SHA1', 'SHA256']:
        for iters in [256000, 64000, 4000]:
            for pw in [PASS_NO_QUOTE, PASS_WITH_QUOTE, RAW_HEX]:
                pw_label = 'no_quote' if pw == PASS_NO_QUOTE else ('with_quote' if pw == PASS_WITH_QUOTE else 'raw')
                label = f"ps={ps} kdf={kdf}/{iters} pw={pw_label}"
                def make_setup(page_size=ps, algo=kdf, iterations=iters, passphrase=pw):
                    def setup(c):
                        # Set params BEFORE key
                        c.execute(f"PRAGMA cipher_page_size = {page_size};")
                        c.execute(f"PRAGMA kdf_iter = {iterations};")
                        c.execute(f"PRAGMA cipher_kdf_algorithm = PBKDF2_HMAC_{algo};")
                        c.execute(f"PRAGMA key = \"{passphrase}\";")
                    return setup
                configs.append((label, make_setup()))

# cipher_compatibility shortcuts (key after)
for compat in [3, 4, 2]:
    for pw in [PASS_NO_QUOTE, PASS_WITH_QUOTE, RAW_HEX]:
        pw_label = 'no_quote' if pw == PASS_NO_QUOTE else ('with_quote' if pw == PASS_WITH_QUOTE else 'raw')
        label = f"compat={compat} pw={pw_label}"
        def make_setup_compat(c_ver=compat, passphrase=pw):
            def setup(c):
                c.execute(f"PRAGMA cipher_compatibility = {c_ver};")
                c.execute(f"PRAGMA key = \"{passphrase}\";")
            return setup
        configs.append((label, make_setup_compat()))

# compat=3 + explicit page size
for ps in [1024, 4096]:
    for pw in [PASS_NO_QUOTE, PASS_WITH_QUOTE]:
        pw_label = 'no_quote' if pw == PASS_NO_QUOTE else 'with_quote'
        label = f"compat=3 ps={ps} pw={pw_label} (key last)"
        def make_setup3(page_size=ps, passphrase=pw):
            def setup(c):
                c.execute(f"PRAGMA cipher_compatibility = 3;")
                c.execute(f"PRAGMA cipher_page_size = {page_size};")
                c.execute(f"PRAGMA key = \"{passphrase}\";")
            return setup
        configs.append((label, make_setup3()))

# no HMAC
for ps in [1024, 4096]:
    for pw in [PASS_NO_QUOTE, PASS_WITH_QUOTE]:
        pw_label = 'no_quote' if pw == PASS_NO_QUOTE else 'with_quote'
        label = f"no_hmac ps={ps} pw={pw_label}"
        def make_setup_nhmac(page_size=ps, passphrase=pw):
            def setup(c):
                c.execute(f"PRAGMA cipher_page_size = {page_size};")
                c.execute("PRAGMA cipher_use_hmac = 0;")
                c.execute(f"PRAGMA key = \"{passphrase}\";")
            return setup
        configs.append((label, make_setup_nhmac()))

if not DB.exists():
    print(f"ERROR: {DB} not found")
    sys.exit(1)

print(f"Testing {len(configs)} combinations on {DB}...")
found = False
for label, setup_fn in configs:
    if try_open(label, setup_fn):
        found = True
        break

if not found:
    print(f"\nAll {len(configs)} combinations failed.")
    # Print file header for analysis
    with DB.open('rb') as f:
        header = f.read(32)
    print(f"File header (32 bytes): {header.hex()}")
    print(f"File size: {DB.stat().st_size:,} bytes")
    print(f"Mod 1024: {DB.stat().st_size % 1024}")
    print(f"Mod 4096: {DB.stat().st_size % 4096}")
