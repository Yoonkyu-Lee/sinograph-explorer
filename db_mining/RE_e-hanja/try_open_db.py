#!/usr/bin/env python3
"""
Systematic SQLCipher key/param test for ejajeon.dat and ehanja.user.dat.

Key derivation facts:
  - App calls sqlite3_key_v2() with UTF-8 bytes of "x'34abb122029c2c34e00046daa11b5c67"
    (no closing quote; 34 bytes)
  - sqlcipher3 Python may handle x'...' as raw hex at PRAGMA level.
  - ejajeon.dat: 347,125,760 bytes = mod(1024)=0 → strong hint page_size=1024
  - ehanja.user.dat: 28,672 bytes = mod(1024)=0, mod(4096)=0
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import sqlcipher3

# Known passphrase the app passes to sqlite3_key_v2
PASS_STR = "x'34abb122029c2c34e00046daa11b5c67"   # no closing quote
PASS_HEX = "x'34abb122029c2c34e00046daa11b5c67'"  # with closing quote (PRAGMA raw hex)
RAW_HEX  = "34abb122029c2c34e00046daa11b5c67"     # just the hex digits

ROOT = Path(__file__).resolve().parent

DB_FILES = [
    ROOT / "ejajeon.dat",
    ROOT / "ehanja.user.dat",
]

def try_open(db_path, setup_fn, label):
    """Try to open db_path, run setup_fn(conn), then verify with SELECT."""
    try:
        conn = sqlcipher3.connect(str(db_path))
        c = conn.cursor()
        setup_fn(c)
        # Verification: try to read schema
        c.execute("SELECT count(*) FROM sqlite_master;")
        row = c.fetchone()
        conn.close()
        print(f"  [SUCCESS] {label} → sqlite_master count={row[0]}")
        return True
    except Exception as e:
        err = str(e).split('\n')[0][:80]
        print(f"  [FAIL] {label}: {err}")
        return False

def run_tests(db_path):
    print(f"\n{'='*60}")
    print(f"DB: {db_path}")
    print(f"{'='*60}")

    configs = []

    # --- Raw hex key (x'...') with various page sizes ---
    for ps in [1024, 2048, 4096, 512]:
        for compat in [None, 3, 4]:
            label_parts = [f"raw_hex page={ps}"]
            if compat:
                label_parts.append(f"compat={compat}")
            label = " ".join(label_parts)
            def make_setup(page_size=ps, compatibility=compat):
                def setup(c):
                    c.execute(f"PRAGMA key = \"{PASS_HEX}\";")
                    c.execute(f"PRAGMA cipher_page_size = {page_size};")
                    if compatibility == 3:
                        c.execute("PRAGMA cipher_compatibility = 3;")
                    elif compatibility == 4:
                        c.execute("PRAGMA cipher_compatibility = 4;")
                return setup
            configs.append((label, make_setup()))

    # --- Passphrase (no closing quote) with various page sizes and KDF ---
    for ps in [1024, 4096]:
        for kdf_iter in [64000, 256000]:
            for kdf_algo in ['sha1', 'sha512']:
                label = f"passphrase(no-close-quote) page={ps} kdf={kdf_algo}/{kdf_iter}"
                def make_setup2(page_size=ps, iters=kdf_iter, algo=kdf_algo):
                    def setup(c):
                        c.execute(f"PRAGMA key = \"{PASS_STR}\";")
                        c.execute(f"PRAGMA cipher_page_size = {page_size};")
                        c.execute(f"PRAGMA kdf_iter = {iters};")
                        c.execute(f"PRAGMA cipher_kdf_algorithm = PBKDF2_HMAC_{algo.upper()};")
                    return setup
                configs.append((label, make_setup2()))

    # --- Full passphrase with closing quote (treated as plain string) ---
    for ps in [1024, 4096]:
        label = f"passphrase(with-close-quote) page={ps}"
        def make_setup3(page_size=ps):
            def setup(c):
                c.execute(f"PRAGMA key = \"{PASS_HEX}\";")
                c.execute(f"PRAGMA cipher_page_size = {page_size};")
                # This might be treated as raw hex by sqlcipher PRAGMA handling
            return setup
        configs.append((label, make_setup3()))

    # --- HMAC disabled ---
    for ps in [1024, 4096]:
        label = f"raw_hex no_hmac page={ps}"
        def make_setup4(page_size=ps):
            def setup(c):
                c.execute(f"PRAGMA key = \"{PASS_HEX}\";")
                c.execute(f"PRAGMA cipher_page_size = {page_size};")
                c.execute("PRAGMA cipher_use_hmac = 0;")
            return setup
        configs.append((label, make_setup4()))

    # --- SQLCipher v3 compat (SHA1/64000/page1024) ---
    label = "v3compat page=1024 explicit"
    def setup_v3_1024(c):
        c.execute(f"PRAGMA key = \"{PASS_HEX}\";")
        c.execute("PRAGMA cipher_compatibility = 3;")
        c.execute("PRAGMA cipher_page_size = 1024;")
    configs.append((label, setup_v3_1024))

    # --- SQLCipher v2 compat ---
    label = "v2compat page=1024"
    def setup_v2(c):
        c.execute(f"PRAGMA key = \"{PASS_HEX}\";")
        c.execute("PRAGMA cipher_compatibility = 2;")
        c.execute("PRAGMA cipher_page_size = 1024;")
    configs.append((label, setup_v2))

    # --- Try just the raw 16-byte hex as plain passphrase ---
    for ps in [1024, 4096]:
        label = f"raw_pass(hex_digits_only) page={ps} sha1/64000"
        def make_setup5(page_size=ps):
            def setup(c):
                c.execute(f"PRAGMA key = \"{RAW_HEX}\";")
                c.execute(f"PRAGMA cipher_page_size = {page_size};")
                c.execute("PRAGMA kdf_iter = 64000;")
                c.execute("PRAGMA cipher_kdf_algorithm = PBKDF2_HMAC_SHA1;")
            return setup
        configs.append((label, make_setup5()))

    found = False
    for label, setup_fn in configs:
        if try_open(db_path, setup_fn, label):
            found = True
            break

    if not found:
        print(f"\n  All {len(configs)} combinations failed.")
    return found

# First, print current sqlcipher3 version info
try:
    conn = sqlcipher3.connect(":memory:")
    c = conn.cursor()
    c.execute("PRAGMA cipher_version;")
    ver = c.fetchone()
    c.execute("PRAGMA cipher_provider;")
    prov = c.fetchone()
    conn.close()
    print(f"sqlcipher3 version: {ver}")
    print(f"sqlcipher3 provider: {prov}")
except Exception as e:
    print(f"sqlcipher3 version check failed: {e}")

for db in DB_FILES:
    if db.exists():
        run_tests(db)
    else:
        print(f"\nSkipping {db} (not found)")
