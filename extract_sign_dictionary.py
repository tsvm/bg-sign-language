#!/usr/bin/env python3
"""
Bulgarian Sign Language Dictionary Extractor
=============================================
Matches images to captions by grouping images into rows (same y-band),
then pairing them left-to-right with the caption lines in order.

Usage:
    pip install pymupdf pillow

    python extract_sign_dictionary.py dictionary.pdf --pages 2-5 --debug
    python extract_sign_dictionary.py dictionary.pdf

Output:
    images/       one image per sign, e.g. 00001_АКАДЕМИЯ_АКАДЕМИЧЕН.jpg
    index.json    searchable index
    index.sqlite  SQLite database
    problems.log  unmatched images
"""

import fitz
import json
import sqlite3
import re
import sys
import logging
from pathlib import Path
from dataclasses import dataclass, asdict

try:
    from PIL import Image, ImageOps
    import io
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("WARNING: pip install pillow  (needed for colour-space fixes)")

# ── Configuration ─────────────────────────────────────────────────────────────

OUTPUT_DIR   = Path("images")
INDEX_JSON   = Path("index.json")
INDEX_SQLITE = Path("index.sqlite")
PROBLEMS_LOG = Path("problems.log")

MIN_IMAGE_WIDTH  = 100   # points – filters decorative images
MIN_IMAGE_HEIGHT = 100

# Two images are in the same "row" if their vertical centres are within this many points
ROW_TOLERANCE = 80

# A caption block belongs to a row if its top is within this many points below the row bottom
CAPTION_GAP = 150

# Pages to skip – update SKIP_PAGES_BEFORE after running find_start.py
SKIP_PAGES_BEFORE = 10
SKIP_PAGES_AFTER  = 3

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Entry:
    entry_number: int | None
    label: str
    label_raw: str
    filename: str
    page: int

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_label(text: str) -> tuple[int | None, str]:
    text = text.strip()
    m = re.match(r'^(\d+)[.\t\s]+(.+)$', text, re.DOTALL)
    if m:
        return int(m.group(1)), m.group(2).strip()
    return None, text


def split_caption_block(block_text: str) -> list[str]:
    """Split '1.\tA\n2.\tB\n3.\tC' into ['1.\tA', '2.\tB', '3.\tC']."""
    lines = [l.strip() for l in block_text.strip().splitlines() if l.strip()]
    merged = []
    for line in lines:
        if re.match(r'^\d+[.\t\s]', line):
            merged.append(line)
        elif merged:
            merged[-1] += ' ' + line
        else:
            merged.append(line)
    return merged


def safe_filename(entry_number: int | None, label: str, suffix: str = "") -> str:
    clean = re.sub(r'[\\/*?:"<>|;\n\t]', '_', label)
    clean = re.sub(r'_+', '_', clean).strip('_')
    clean = clean[:80]
    num = f"{entry_number:05d}" if entry_number is not None else "NOID"
    return f"{num}_{clean}{suffix}.jpg"


def is_content_image(rect: fitz.Rect) -> bool:
    return rect.width >= MIN_IMAGE_WIDTH and rect.height >= MIN_IMAGE_HEIGHT


def fix_image(img_dict: dict) -> bytes:
    """Convert to clean RGB JPEG, fixing inverted/CMYK/DeviceN images."""
    raw = img_dict["image"]
    if not HAS_PIL:
        return raw
    try:
        img = Image.open(io.BytesIO(raw))

        # Normalise mode to RGB first
        if img.mode == "CMYK":
            img = img.convert("RGB")
        elif img.mode == "P":
            img = img.convert("RGB")
        elif img.mode not in ("RGB", "L", "RGBA"):
            img = img.convert("RGB")

        # Invert if the image looks like a negative.
        # Normal photos: avg brightness 140+
        # DeviceN black-plate images: avg ~90  (ink density, not brightness)
        gray = img.convert("L")
        avg = sum(gray.getdata()) / (gray.width * gray.height)
        if avg < 110:
            img = ImageOps.invert(img.convert("L")).convert("RGB")

        if img.mode != "RGB":
            img = img.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return buf.getvalue()
    except Exception as e:
        logging.warning(f"Image conversion failed: {e}")
        return raw


def group_into_rows(images: list[tuple[fitz.Rect, int]]) -> list[list[tuple[fitz.Rect, int]]]:
    """
    Group images into horizontal rows based on vertical centre proximity.
    Within each row, sort left-to-right by x-centre.
    """
    if not images:
        return []

    # Sort by vertical centre
    by_cy = sorted(images, key=lambda t: (t[0].y0 + t[0].y1) / 2)

    rows = []
    current_row = [by_cy[0]]
    current_cy = (by_cy[0][0].y0 + by_cy[0][0].y1) / 2

    for img in by_cy[1:]:
        cy = (img[0].y0 + img[0].y1) / 2
        if abs(cy - current_cy) <= ROW_TOLERANCE:
            current_row.append(img)
        else:
            rows.append(current_row)
            current_row = [img]
            current_cy = cy

    rows.append(current_row)

    # Sort each row left-to-right
    for row in rows:
        row.sort(key=lambda t: (t[0].x0 + t[0].x1) / 2)

    return rows


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_page(page: fitz.Page, doc: fitz.Document, page_num: int,
                 debug: bool = False) -> list[Entry]:
    entries = []

    # 1. Collect content images
    images = []
    seen_xrefs = set()
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        rects = page.get_image_rects(xref)
        if not rects:
            continue
        rect = rects[0]
        if not is_content_image(rect):
            continue
        images.append((rect, xref))

    if not images:
        return entries

    # 2. Parse caption blocks
    caption_blocks = []   # (y_top, raw_block_text)
    for b in page.get_text("blocks"):
        x0, y0, x1, y1, text, _, btype = b
        if btype != 0 or not text.strip():
            continue
        if re.match(r'^\d{1,3}$', text.strip()):
            continue
        if "Речник на българския жестов език" in text:
            continue
        caption_blocks.append((y0, text))

    # 3. Group images into rows
    rows = group_into_rows(images)

    if debug:
        print(f"\n--- Page {page_num+1}: {len(images)} images in {len(rows)} rows ---")
        for i, row in enumerate(rows):
            cxs = [f"cx={((r.x0+r.x1)/2):.0f}" for r, _ in row]
            row_bottom = max(r.y1 for r, _ in row)
            print(f"  Row {i+1} (bottom={row_bottom:.0f}): {len(row)} images  {cxs}")

    # 4. For each row, find the caption block just below it and pair left-to-right
    for row in rows:
        row_top    = min(r.y0 for r, _ in row)
        row_bottom = max(r.y1 for r, _ in row)

        # Find caption blocks whose top is just below this row's bottom
        matching_blocks = [
            (y0, text) for (y0, text) in caption_blocks
            if row_bottom < y0 < row_bottom + CAPTION_GAP
        ]

        if not matching_blocks:
            for rect, xref in row:
                logging.warning(f"Page {page_num+1}: no caption block for row at y={row_top:.0f}-{row_bottom:.0f}")
            if debug:
                print(f"  [NO CAPTION] for row bottom={row_bottom:.0f}")
            continue

        # Collect all individual label lines from all matching blocks
        all_lines = []
        for _, block_text in matching_blocks:
            all_lines.extend(split_caption_block(block_text))

        # Sort lines by entry number so order is deterministic
        def line_sort_key(line):
            m = re.match(r'^(\d+)', line.strip())
            return int(m.group(1)) if m else 9999

        all_lines.sort(key=line_sort_key)

        if debug:
            print(f"  Caption lines ({len(all_lines)}): {[l[:40] for l in all_lines]}")

        if len(all_lines) != len(row):
            logging.warning(
                f"Page {page_num+1}: row has {len(row)} images but {len(all_lines)} caption lines"
                f" — pairing by position anyway"
            )

        # Pair: leftmost image → first line, etc.
        for i, (rect, xref) in enumerate(row):
            if i >= len(all_lines):
                logging.warning(f"Page {page_num+1}: more images than captions in row")
                break

            label_raw = all_lines[i]
            entry_number, label = parse_label(label_raw)
            filename = safe_filename(entry_number, label)

            out_path = OUTPUT_DIR / filename
            collision = 1
            while out_path.exists():
                collision += 1
                filename = safe_filename(entry_number, label, suffix=f"_v{collision}")
                out_path = OUTPUT_DIR / filename

            try:
                img_dict = doc.extract_image(xref)
                img_bytes = fix_image(img_dict)
                out_path.write_bytes(img_bytes)
            except Exception as e:
                logging.warning(f"Page {page_num+1}: could not extract xref={xref}: {e}")
                continue

            if debug:
                print(f"  → {filename}")

            entries.append(Entry(
                entry_number=entry_number,
                label=label,
                label_raw=label_raw,
                filename=str(out_path),
                page=page_num + 1,
            ))

    return entries

# ── SQLite ────────────────────────────────────────────────────────────────────

def build_sqlite(entries: list[Entry]):
    conn = sqlite3.connect(INDEX_SQLITE)
    conn.execute("DROP TABLE IF EXISTS signs")
    conn.execute("""
        CREATE TABLE signs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_number INTEGER,
            label TEXT,
            label_raw TEXT,
            filename TEXT,
            page INTEGER
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_entry ON signs(entry_number)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_label ON signs(label)")
    for e in entries:
        conn.execute(
            "INSERT INTO signs VALUES (NULL,?,?,?,?,?)",
            (e.entry_number, e.label, e.label_raw, e.filename, e.page)
        )
    conn.commit()
    conn.close()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("--pages", type=str, default=None,
                        help="1-indexed range e.g. '2-10'")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found"); sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    logging.basicConfig(filename=PROBLEMS_LOG, level=logging.WARNING, format="%(message)s")

    print(f"Opening {pdf_path} ...")
    doc = fitz.open(str(pdf_path))
    total = len(doc)
    print(f"Total pages: {total}")

    if args.pages:
        s, e = args.pages.split("-")
        page_range = range(int(s) - 1, int(e))
    else:
        page_range = range(SKIP_PAGES_BEFORE, total - SKIP_PAGES_AFTER)

    all_entries: list[Entry] = []

    for i, page_num in enumerate(page_range):
        entries = extract_page(doc[page_num], doc, page_num, debug=args.debug)
        all_entries.extend(entries)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(page_range)} pages, {len(all_entries)} entries...")

    doc.close()
    print(f"\nExtracted {len(all_entries)} entries")

    all_entries.sort(key=lambda e: (e.entry_number or 0, e.label))

    with open(INDEX_JSON, "w", encoding="utf-8") as f:
        json.dump([asdict(e) for e in all_entries], f, ensure_ascii=False, indent=2)
    print(f"JSON   → {INDEX_JSON}")

    build_sqlite(all_entries)
    print(f"SQLite → {INDEX_SQLITE}")
    print(f"Search: sqlite3 {INDEX_SQLITE} \"SELECT * FROM signs WHERE label LIKE '%МАЙКА%'\"")

    no_number = [e for e in all_entries if e.entry_number is None]
    if no_number:
        print(f"\nWARNING: {len(no_number)} entries without entry number — see {PROBLEMS_LOG}")
    else:
        print("All entries have numbers. Looks clean!")


if __name__ == "__main__":
    main()
