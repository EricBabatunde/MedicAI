#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
  Medical PDF → Structured JSON Extraction Pipeline
  ---------------------------------------------------
  Engine:   PyMuPDF (fitz) for text + font analysis
            pdfplumber for table regions (dynamic handoff)
  Chunking: LangChain MarkdownHeaderTextSplitter
  Output:   JSON array matching res/JSON_schema.json
═══════════════════════════════════════════════════════════════════════════
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
from langchain_text_splitters import MarkdownHeaderTextSplitter


# ─────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────

# Domain spoke keyword map — maps keywords found in text to a domain tag
DOMAIN_KEYWORDS = {
    "surgery": "minor_surgery",
    "surgical": "minor_surgery",
    "incision": "minor_surgery",
    "suture": "minor_surgery",
    "obstetric": "obstetrics",
    "pregnancy": "obstetrics",
    "labour": "obstetrics",
    "labor": "obstetrics",
    "delivery": "obstetrics",
    "newborn": "neonatology",
    "neonatal": "neonatology",
    "malaria": "infectious_disease",
    "tuberculosis": "infectious_disease",
    "hiv": "infectious_disease",
    "hepatitis": "infectious_disease",
    "antibiotic": "pharmacology",
    "dosing": "pharmacology",
    "drug": "pharmacology",
    "formulary": "pharmacology",
    "trauma": "emergency_medicine",
    "shock": "emergency_medicine",
    "resuscitation": "emergency_medicine",
    "burn": "emergency_medicine",
    "fracture": "orthopaedics",
    "splint": "orthopaedics",
    "cast": "orthopaedics",
    "diabetes": "chronic_care",
    "hypertension": "chronic_care",
    "asthma": "chronic_care",
    "copd": "chronic_care",
    "heart failure": "chronic_care",
    "epilepsy": "chronic_care",
    "mental": "mental_health",
    "depression": "mental_health",
    "psychosis": "mental_health",
    "dermatology": "dermatology",
    "skin": "dermatology",
    "nutrition": "nutrition",
    "malnutrition": "nutrition",
    "anaesthesia": "anaesthesia",
    "anesthesia": "anaesthesia",
    "eye": "ophthalmology",
    "dental": "dental",
    "ear": "ent",
    "nose": "ent",
    "throat": "ent",
}

# Clinical category keyword map
CATEGORY_KEYWORDS = {
    "treatment": [
        "treat", "therapy", "administer", "prescribe", "dose", "dosing",
        "give", "medication", "drug", "antibiotic", "mg", "ml",
    ],
    "procedure": [
        "incision", "suture", "drainage", "insert", "remove", "perform",
        "technique", "step", "procedure", "surgical", "cannula", "catheter",
    ],
    "diagnosis": [
        "diagnos", "differential", "sign", "symptom", "clinical features",
        "assessment", "examination", "classification", "criteria", "suspect",
    ],
}


# ─────────────────────────────────────────────────────────────────────────
#  Font-Size to Markdown Header Mapping (per-document adaptive)
# ─────────────────────────────────────────────────────────────────────────

def build_font_size_map(doc):
    """
    Scan first 40 pages to build an adaptive font-size → heading-level map.
    Returns a sorted list of (min_size, markdown_prefix) tuples, largest first.
    """
    from collections import Counter
    size_counter = Counter()

    scan_limit = min(40, doc.page_count)
    for pnum in range(scan_limit):
        page = doc[pnum]
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    size = round(span["size"], 1)
                    text = span["text"].strip()
                    if text:
                        size_counter[size] += 1

    # Body text is the most frequent size
    body_size = size_counter.most_common(1)[0][0] if size_counter else 10.0

    # Collect sizes larger than body text, sorted descending
    heading_sizes = sorted(
        [s for s in size_counter if s > body_size + 1.0],
        reverse=True,
    )

    # Map to heading levels: largest → #, next → ##, etc (max ####)
    size_map = {}
    for i, s in enumerate(heading_sizes[:4]):
        level = i + 1
        size_map[s] = "#" * level
    size_map["body"] = body_size

    print(f"  📐 Body text size: {body_size}pt")
    for s, prefix in size_map.items():
        if s != "body":
            print(f"  📐 {prefix} heading: ≥{s}pt")

    return size_map


# ─────────────────────────────────────────────────────────────────────────
#  Regex Cleaning
# ─────────────────────────────────────────────────────────────────────────

# Pre-compiled patterns for performance
RE_PAGE_NUM = re.compile(
    r"(?m)"
    r"(?:^[\s]*Page\s+\d+\s*/\s*\d+[\s]*$)"     # "Page 11 / 409"
    r"|(?:^[\s]*\d{1,4}[\s]*$)"                   # Bare page numbers
    r"|(?:^[\s]*-\s*\d+\s*-[\s]*$)"               # "- 12 -"
)
RE_HEADER_FOOTER = re.compile(
    r"(?m)"
    r"(?:^[\s]*(?:Chapter|CHAPTER)\s+\d+.*$)"     # "Chapter 1 ..." headers
    r"|(?:^[\s]*(?:WHO|MSF|ATLS|©).*$)"           # Org headers / copyright lines
    r"|(?:^[\s]*Last updated:.*$)"                 # "Last updated: September 2023"
)
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def clean_text(raw_text):
    """Strip page numbers, header/footer artifacts, URLs, and normalise whitespace."""
    text = RE_PAGE_NUM.sub("", raw_text)
    text = RE_HEADER_FOOTER.sub("", text)
    text = RE_URL.sub("", text)
    text = RE_MULTI_SPACE.sub(" ", text)
    text = RE_MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────
#  Table Extraction (pdfplumber handoff)
# ─────────────────────────────────────────────────────────────────────────

def extract_tables_plumber(pdf_path, page_num, table_bboxes):
    """
    Use pdfplumber to extract table data for specific bounding boxes
    detected by PyMuPDF on a given page.
    Returns a list of tables, each table being a list of row-lists.
    """
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        if page_num >= len(pdf.pages):
            return tables
        plumber_page = pdf.pages[page_num]

        for bbox in table_bboxes:
            # pdfplumber uses (x0, top, x1, bottom) — same as fitz bbox
            try:
                cropped = plumber_page.crop(bbox)
                table = cropped.extract_table()
                if table:
                    # Clean None values and strip whitespace
                    cleaned = [
                        [(cell or "").strip() for cell in row]
                        for row in table
                    ]
                    tables.append(cleaned)
            except Exception as e:
                print(f"    ⚠️  pdfplumber table extraction failed for bbox {bbox}: {e}")

    return tables


# ─────────────────────────────────────────────────────────────────────────
#  Page-by-Page Extraction (PyMuPDF primary, pdfplumber for tables)
# ─────────────────────────────────────────────────────────────────────────

def extract_page(doc, page_num, pdf_path, font_size_map):
    """
    Extract text and tables from a single page.
    Returns (markdown_text: str, tables: list, had_tables: bool)
    """
    page = doc[page_num]
    body_size = font_size_map.get("body", 10.0)

    # ── Detect tables via PyMuPDF ────────────────────────────────────
    fitz_tables = page.find_tables()
    table_bboxes = [t.bbox for t in fitz_tables.tables] if fitz_tables.tables else []
    table_rects = [fitz.Rect(b) for b in table_bboxes]

    # ── Hand table regions to pdfplumber ─────────────────────────────
    extracted_tables = []
    if table_bboxes:
        print(f"    🔀 Handing {len(table_bboxes)} table region(s) to pdfplumber...")
        extracted_tables = extract_tables_plumber(pdf_path, page_num, table_bboxes)
        print(f"    ✅ pdfplumber extracted {len(extracted_tables)} table(s)")

    # ── Extract text blocks via PyMuPDF (skip table regions) ─────────
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    md_lines = []

    for block in blocks:
        if block["type"] != 0:  # skip image blocks
            continue

        block_rect = fitz.Rect(block["bbox"])

        # Skip blocks that fall inside a detected table region
        in_table = any(
            block_rect.intersects(tr) and block_rect.get_area() > 0
            and (block_rect & tr).get_area() / block_rect.get_area() > 0.5
            for tr in table_rects
        )
        if in_table:
            continue

        for line in block["lines"]:
            line_text_parts = []
            line_max_size = 0
            line_is_bold = False

            for span in line["spans"]:
                text = span["text"]
                size = round(span["size"], 1)
                font = span.get("font", "")

                line_text_parts.append(text)
                line_max_size = max(line_max_size, size)
                if "Bold" in font or "bold" in font:
                    line_is_bold = True

            line_text = "".join(line_text_parts).strip()
            if not line_text:
                continue

            # ── Map font size to markdown heading ────────────────────
            heading_prefix = None
            for size_threshold, prefix in font_size_map.items():
                if size_threshold == "body":
                    continue
                if line_max_size >= size_threshold:
                    heading_prefix = prefix
                    break  # Already sorted largest-first from build step

            if heading_prefix:
                md_lines.append(f"\n{heading_prefix} {line_text}\n")
            elif line_is_bold and line_max_size > body_size:
                # Bold text slightly above body = sub-heading (####)
                md_lines.append(f"\n#### {line_text}\n")
            else:
                md_lines.append(line_text)

    markdown_text = "\n".join(md_lines)
    return markdown_text, extracted_tables, bool(table_bboxes)


# ─────────────────────────────────────────────────────────────────────────
#  Classification Helpers
# ─────────────────────────────────────────────────────────────────────────

def classify_domain(text):
    """Assign a domain spoke based on keyword frequency in the text."""
    text_lower = text.lower()
    scores = {}
    for keyword, domain in DOMAIN_KEYWORDS.items():
        count = text_lower.count(keyword)
        if count > 0:
            scores[domain] = scores.get(domain, 0) + count

    if not scores:
        return "general_medicine"
    return max(scores, key=scores.get)


def classify_category(text):
    """Assign 'treatment', 'diagnosis', or 'procedure' via keyword matching."""
    text_lower = text.lower()
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(text_lower.count(kw) for kw in keywords)
        scores[category] = score

    if not any(scores.values()):
        return "diagnosis"  # default
    return max(scores, key=scores.get)


def generate_chunk_id(source_name, chapter, topic, index):
    """Generate a short, unique hash-based chunk ID."""
    raw = f"{source_name}|{chapter}|{topic}|{index}"
    h = hashlib.sha256(raw.encode()).hexdigest()[:10]
    # Build a readable prefix from source
    prefix = re.sub(r"[^a-z0-9]", "_", source_name.lower())[:20]
    return f"{prefix}_{h}"


# ─────────────────────────────────────────────────────────────────────────
#  Main Pipeline
# ─────────────────────────────────────────────────────────────────────────

def process_pdf(pdf_path, output_path):
    """Full extraction pipeline for a single PDF."""
    pdf_path = str(pdf_path)
    source_name = Path(pdf_path).stem

    print(f"\n{'═' * 70}")
    print(f"  📄 Processing: {Path(pdf_path).name}")
    print(f"{'═' * 70}")

    # ── Open with PyMuPDF ────────────────────────────────────────────
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    print(f"  📊 Total pages: {total_pages}")

    # ── Build adaptive font-size → heading map ───────────────────────
    print(f"\n  🔍 Scanning font sizes for heading detection...")
    font_size_map = build_font_size_map(doc)

    # ── Page-by-page extraction ──────────────────────────────────────
    full_markdown = ""
    all_tables = {}        # page_num → list of tables
    page_table_count = 0

    print(f"\n  📖 Extracting text page-by-page...\n")

    for page_num in range(total_pages):
        if (page_num + 1) % 25 == 0 or page_num == 0:
            print(f"    ⏳ Page {page_num + 1}/{total_pages}...")

        page_md, page_tables, had_tables = extract_page(
            doc, page_num, pdf_path, font_size_map
        )

        if had_tables:
            page_table_count += 1

        # Clean the extracted text
        cleaned = clean_text(page_md)

        if cleaned:
            # Inject a page reference marker for later mapping
            full_markdown += f"\n<!-- PAGE:{page_num + 1} -->\n{cleaned}\n"

        if page_tables:
            all_tables[page_num + 1] = page_tables

    doc.close()
    print(f"\n  ✅ Extraction complete: {len(full_markdown):,} chars, {page_table_count} pages with tables")

    # ── Semantic chunking with LangChain ─────────────────────────────
    print(f"\n  ✂️  Chunking with MarkdownHeaderTextSplitter...")

    headers_to_split_on = [
        ("#", "chapter"),
        ("##", "primary_topic"),
        ("###", "sub_topic"),
        ("####", "sub_sub_topic"),
    ]
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    chunks = splitter.split_text(full_markdown)
    print(f"  ✅ Generated {len(chunks)} semantic chunks")

    # ── Map chunks to output schema ──────────────────────────────────
    print(f"\n  🏗️  Building structured JSON records...")

    output_records = []
    for i, chunk in enumerate(chunks):
        text = chunk.page_content
        metadata = chunk.metadata

        # Skip tiny fragments (likely artefacts)
        if len(text.strip()) < 30:
            continue

        # Extract page references from embedded markers
        page_refs = sorted(set(
            int(m) for m in re.findall(r"<!-- PAGE:(\d+) -->", text)
        ))
        # Strip the markers from the final text
        clean_content = re.sub(r"<!-- PAGE:\d+ -->\s*", "", text).strip()

        if not clean_content:
            continue

        # Gather tables from referenced pages
        chunk_tables = []
        for pg in page_refs:
            if pg in all_tables:
                chunk_tables.extend(all_tables[pg])

        # Build hierarchical context
        chapter = metadata.get("chapter", "")
        primary_topic = metadata.get("primary_topic", "")
        sub_topic = metadata.get("sub_topic", "")

        # Classification
        domain = classify_domain(clean_content)
        category = classify_category(clean_content)
        chunk_id = generate_chunk_id(source_name, chapter, primary_topic, i)

        record = {
            "chunk_id": chunk_id,
            "domain_spoke": domain,
            "source_text": source_name.replace("_", " "),
            "hierarchical_context": {
                "chapter": chapter,
                "primary_topic": primary_topic,
                "sub_topic": sub_topic,
            },
            "clinical_category": category,
            "text_content": clean_content,
            "extracted_tables": chunk_tables,
            "page_reference": page_refs,
        }
        output_records.append(record)

        if (i + 1) % 50 == 0:
            print(f"    📦 Mapped {i + 1}/{len(chunks)} chunks...")

    print(f"  ✅ Final record count: {len(output_records)} (dropped {len(chunks) - len(output_records)} tiny fragments)")

    # ── Write output ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_records, f, indent=2, ensure_ascii=False)

    print(f"\n  💾 Written to: {output_path}")
    print(f"  📊 File size: {os.path.getsize(output_path) / 1024:.1f} KB")

    return output_records


# ─────────────────────────────────────────────────────────────────────────
#  CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract medical PDFs into structured JSON chunks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single PDF:
  python scripts/extract_pdf.py res/sources_pdf/MSF_clinical_guidelines.pdf

  # Process all PDFs in the sources directory:
  python scripts/extract_pdf.py res/sources_pdf/

  # Specify custom output path:
  python scripts/extract_pdf.py res/sources_pdf/MSF_clinical_guidelines.pdf -o data/msf_chunks.json
        """,
    )
    parser.add_argument(
        "input",
        help="Path to a single PDF file or a directory containing PDFs.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output JSON file path (default: data/<source_name>_chunks.json).",
    )

    args = parser.parse_args()
    input_path = Path(args.input)

    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        pdf_files = [input_path]
    elif input_path.is_dir():
        pdf_files = sorted(input_path.glob("*.pdf"))
        if not pdf_files:
            print(f"❌ No PDF files found in: {input_path}")
            sys.exit(1)
        print(f"📁 Found {len(pdf_files)} PDF(s) in {input_path}")
    else:
        print(f"❌ Invalid input: {input_path}")
        sys.exit(1)

    all_records = []
    for pdf_file in pdf_files:
        if args.output and len(pdf_files) == 1:
            out_path = args.output
        else:
            out_path = f"data/{pdf_file.stem}_chunks.json"

        records = process_pdf(pdf_file, out_path)
        all_records.extend(records)

    # If processing multiple PDFs, also write a combined output
    if len(pdf_files) > 1:
        combined_path = args.output or "data/all_extracted_chunks.json"
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(all_records, f, indent=2, ensure_ascii=False)
        print(f"\n{'═' * 70}")
        print(f"  📦 Combined output: {combined_path}")
        print(f"  📊 Total chunks across all PDFs: {len(all_records)}")
        print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
