from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference.upper())
    if not letters:
        return 0
    value = 0
    for char in letters.group(0):
        value = value * 26 + ord(char) - ord("A") + 1
    return value - 1


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return ["".join(node.text or "" for node in item.iter(f"{{{MAIN_NS}}}t")) for item in root]


def _first_sheet_path(archive: zipfile.ZipFile) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    sheet = workbook.find(f".//{{{MAIN_NS}}}sheet")
    if sheet is None:
        raise ValueError("Excel workbook has no worksheets")
    relation_id = sheet.attrib[f"{{{DOC_REL_NS}}}id"]
    relations = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    for relation in relations.findall(f"{{{REL_NS}}}Relationship"):
        if relation.attrib.get("Id") == relation_id:
            target = relation.attrib["Target"].lstrip("/")
            return target if target.startswith("xl/") else f"xl/{target}"
    raise ValueError("Cannot resolve first worksheet")


def read_xlsx_rows(path: Path) -> list[dict[str, str]]:
    """Read the first worksheet using only the Python standard library."""
    with zipfile.ZipFile(path) as archive:
        shared = _shared_strings(archive)
        sheet = ET.fromstring(archive.read(_first_sheet_path(archive)))

    rows: list[list[str]] = []
    for row in sheet.findall(f".//{{{MAIN_NS}}}row"):
        values: list[str] = []
        for cell in row.findall(f"{{{MAIN_NS}}}c"):
            index = _column_index(cell.attrib.get("r", "A1"))
            while len(values) <= index:
                values.append("")
            cell_type = cell.attrib.get("t", "")
            if cell_type == "inlineStr":
                value = "".join(node.text or "" for node in cell.iter(f"{{{MAIN_NS}}}t"))
            else:
                node = cell.find(f"{{{MAIN_NS}}}v")
                value = node.text if node is not None and node.text is not None else ""
                if cell_type == "s" and value:
                    value = shared[int(value)]
            values[index] = value.strip()
        rows.append(values)

    if not rows:
        return []
    headers = [value.strip() for value in rows[0]]
    records: list[dict[str, str]] = []
    for values in rows[1:]:
        record = {header: values[index].strip() if index < len(values) else "" for index, header in enumerate(headers) if header}
        if any(record.values()):
            records.append(record)
    return records
