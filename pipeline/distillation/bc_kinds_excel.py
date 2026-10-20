import csv
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

INPUT = Path(__file__).parent / "full_data_with_build_success.csv"
OUTPUT = Path(__file__).parent / "full_data_with_build_success_processed.xlsx"


def column_letter(idx: int) -> str:
    """Convert zero-based column index to Excel column letters."""
    letters = []
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        letters.append(chr(65 + rem))
    return "".join(reversed(letters))


def load_rows(path: Path):
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if "BC_kinds" not in reader.fieldnames:
        raise ValueError("BC_kinds column missing")
    return reader.fieldnames, rows


def add_counts_and_sort(header, rows):
    counts = Counter((row.get("BC_kinds", "") or "").strip() for row in rows)
    new_header = ["BC_kinds", "BC_kinds_total"] + [h for h in header if h != "BC_kinds"]

    grouped = defaultdict(list)
    for row in rows:
        kind = (row.get("BC_kinds", "") or "").strip()
        grouped[kind].append(row)

    def safe_int(val):
        try:
            return int(val)
        except (TypeError, ValueError):
            return 0

    # Determine group order: BC_kinds_count ascending, then total desc
    ordered_kinds = sorted(
        grouped.keys(),
        key=lambda k: (
            min(safe_int(r.get("BC_kinds_count", 0)) for r in grouped[k]),
            -counts[k],
        ),
    )

    processed_rows = []
    for kind in ordered_kinds:
        total = counts[kind]
        rows_in_group = sorted(grouped[kind], key=lambda r: safe_int(r.get("BC_kinds_count", 0)))
        for row in rows_in_group:
            new_row = [kind, str(total)]
            for h in header:
                if h == "BC_kinds":
                    continue
                new_row.append(row.get(h, ""))
            processed_rows.append(new_row)

    return new_header, processed_rows


def build_sheet_xml(header, rows):
    def cell_xml(r_idx, c_idx, value):
        col = column_letter(c_idx)
        ref = f"{col}{r_idx}"
        if value is None:
            value = ""
        # Try numeric
        try:
            num = float(value)
            if num.is_integer():
                num = int(num)
            return f'<c r="{ref}" t="n"><v>{num}</v></c>'
        except ValueError:
            text = escape(str(value))
            return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'

    lines = ["<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
             '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
             "<sheetData>"]

    # Header row index starts at 1
    for row_idx, row_values in enumerate([header] + rows, start=1):
        cells = [cell_xml(row_idx, c_idx, val) for c_idx, val in enumerate(row_values)]
        lines.append(f'<row r="{row_idx}">{"".join(cells)}</row>')
    lines.append("</sheetData></worksheet>")
    return "\n".join(lines)


def write_xlsx(header, rows, out_path: Path):
    sheet_xml = build_sheet_xml(header, rows)

    content_types = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">
  <Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>
  <Default Extension=\"xml\" ContentType=\"application/xml\"/>
  <Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/>
  <Override PartName=\"/xl/worksheets/sheet1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>
  <Override PartName=\"/xl/styles.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml\"/>
</Types>"""

    rels_root = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">
  <Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"xl/workbook.xml\"/>
</Relationships>"""

    workbook = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">
  <sheets>
    <sheet name=\"Sheet1\" sheetId=\"1\" r:id=\"rId1\"/>
  </sheets>
</workbook>"""

    workbook_rels = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<Relationships xmlns=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">
  <Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet1.xml\"/>
  <Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles\" Target=\"styles.xml\"/>
</Relationships>"""

    styles = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<styleSheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">
  <fonts count=\"1\"><font/></fonts>
  <fills count=\"1\"><fill/></fills>
  <borders count=\"1\"><border/></borders>
  <cellStyleXfs count=\"1\"><xf/></cellStyleXfs>
  <cellXfs count=\"1\"><xf xfId=\"0\"/></cellXfs>
  <cellStyles count=\"1\"><cellStyle name=\"Normal\" xfId=\"0\" builtinId=\"0\"/></cellStyles>
</styleSheet>"""

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels_root)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)
        zf.writestr("xl/styles.xml", styles)


def main():
    header, rows = load_rows(INPUT)
    new_header, new_rows = add_counts_and_sort(header, rows)
    write_xlsx(new_header, new_rows, OUTPUT)
    print(f"Written: {OUTPUT}")


if __name__ == "__main__":
    main()
