from __future__ import annotations

import csv
import sqlite3
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "ejajeon_plain.db"
CSV_DIR = ROOT / "ejajeon_csv"
PNG_ZIP_PATH = ROOT / "ejajeon_imgData_png.zip"

TABULAR_EXPORTS = (
    ("ftsHub", "ftsHub.csv"),
    ("ftsNatja", "ftsNatja.csv"),
    ("ftsWord", "ftsWord.csv"),
)


def codepoint_label(ch: str) -> str:
    return f"U+{ord(ch):04X}"


def export_query_to_csv(conn: sqlite3.Connection, query: str, output_path: Path) -> int:
    cursor = conn.execute(query)
    columns = [desc[0] for desc in cursor.description]
    row_count = 0

    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in cursor:
            writer.writerow(row)
            row_count += 1

    return row_count


def export_imgdata(conn: sqlite3.Connection) -> tuple[int, Path, Path]:
    manifest_path = CSV_DIR / "imgData_manifest.csv"
    image_count = 0

    with manifest_path.open("w", encoding="utf-8-sig", newline="") as manifest_handle:
        writer = csv.writer(manifest_handle)
        writer.writerow(["hanja", "codepoint", "stroke", "png_size", "zip_path"])

        with zipfile.ZipFile(PNG_ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            cursor = conn.execute(
                "SELECT hanja, stroke, img FROM imgData ORDER BY hanja, stroke"
            )
            for hanja, stroke, blob in cursor:
                cp = codepoint_label(hanja)
                zip_name = f"{cp}/{int(stroke):03d}.png"
                archive.writestr(zip_name, blob)
                writer.writerow([hanja, cp, stroke, len(blob), zip_name])
                image_count += 1

    return image_count, manifest_path, PNG_ZIP_PATH


def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit(f"Database not found: {DB_PATH}")

    CSV_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    try:
        print(f"Database : {DB_PATH}")
        print(f"Output dir: {CSV_DIR}")
        print()

        for object_name, filename in TABULAR_EXPORTS:
            output_path = CSV_DIR / filename
            count = export_query_to_csv(conn, f"SELECT * FROM {object_name}", output_path)
            print(f"Exported {object_name:<8} -> {output_path.name:<20} ({count:,} rows)")

        image_count, manifest_path, zip_path = export_imgdata(conn)
        print(
            f"Exported imgData  -> {manifest_path.name:<20} ({image_count:,} rows)"
        )
        print(
            f"Packed  imgData   -> {zip_path.name:<20} ({zip_path.stat().st_size:,} bytes)"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
