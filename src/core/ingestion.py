import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz
from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from src.core.chunking import structural_chunking
from src.core.document_registry import DEFAULT_CACHE_ROOT
from src.database.vector_store import LegalVectorDB

SUPPORTED_INPUT_EXTENSIONS = {".pdf", ".docx", ".txt"}
WD_FORMAT_PDF = 17
PROVENANCE_SCHEMA_VERSION = 1
UNICODE_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\arialuni.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    r"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ensure_cache_dirs(cache_root: Path) -> dict[str, Path]:
    directories = {
        "root": cache_root,
        "normalized": cache_root / "normalized_pdf",
        "text": cache_root / "extracted_text",
        "records": cache_root / "pdf_records",
        "vectors": cache_root / "vector_cache",
        "chunks": cache_root / "chunk_cache",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def _find_unicode_font_file() -> str | None:
    for candidate in UNICODE_FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def _font_kwargs(font_size: float) -> dict[str, Any]:
    font_file = _find_unicode_font_file()
    if font_file:
        return {
            "fontname": "unicode_fallback",
            "fontfile": font_file,
            "fontsize": font_size,
        }
    return {
        "fontname": "helv",
        "fontsize": font_size,
    }


def _estimate_text_width(text: str, font_size: float) -> float:
    return max(0.0, len(text) * font_size * 0.52)


def compute_file_hash(file_path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        while True:
            data = handle.read(chunk_size)
            if not data:
                break
            digest.update(data)
    return digest.hexdigest()


def _json_dump(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _json_load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return default


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def _normalize_pdf_line_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _copy_to_canonical_pdf(source_path: Path, destination_pdf: Path) -> str:
    if not _is_nonempty_file(destination_pdf):
        shutil.copy2(source_path, destination_pdf)
    return "pdf_passthrough"


def _find_soffice_executable() -> str | None:
    candidates = [
        shutil.which("soffice"),
        shutil.which("libreoffice"),
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def _run_word_export_via_powershell(source_path: Path, destination_pdf: Path) -> None:
    escaped_source = str(source_path).replace("'", "''")
    escaped_destination = str(destination_pdf).replace("'", "''")
    script = f"""
$ErrorActionPreference = 'Stop'
$src = '{escaped_source}'
$dst = '{escaped_destination}'
$word = $null
$doc = $null
try {{
  $word = New-Object -ComObject Word.Application
  $word.Visible = $false
  $doc = $word.Documents.Open($src, $false, $true)
  $doc.ExportAsFixedFormat($dst, {WD_FORMAT_PDF})
}} finally {{
  if ($doc -ne $null) {{ $doc.Close([ref]$false) | Out-Null }}
  if ($word -ne $null) {{ $word.Quit() | Out-Null }}
}}
"""
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not destination_pdf.exists():
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(stderr or "Word COM export to PDF failed.")


def _run_libreoffice_export(source_path: Path, destination_pdf: Path) -> None:
    soffice = _find_soffice_executable()
    if not soffice:
        raise RuntimeError("LibreOffice executable not found.")

    with tempfile.TemporaryDirectory() as temp_dir:
        completed = subprocess.run(
            [
                soffice,
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                temp_dir,
                str(source_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        generated_pdf = Path(temp_dir) / f"{source_path.stem}.pdf"
        if completed.returncode != 0 or not generated_pdf.exists():
            stderr = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(stderr or "LibreOffice export to PDF failed.")
        shutil.move(str(generated_pdf), destination_pdf)


def _iter_block_items(document: DocxDocument) -> Iterable[Paragraph | Table]:
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def _paragraph_has_page_break(paragraph: Paragraph) -> bool:
    for run in paragraph.runs:
        xml = run._element.xml
        if 'w:type="page"' in xml or "lastRenderedPageBreak" in xml:
            return True
    return False


def _length_to_points(value, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(getattr(value, "pt", value) or default)


def _new_pdf_page(pdf_doc: fitz.Document):
    page = pdf_doc.new_page()
    return page, 56


def _ensure_page_space(
    pdf_doc: fitz.Document, page, cursor_y: float, needed_height: float
):
    usable_bottom = page.rect.height - 56
    if cursor_y + needed_height <= usable_bottom:
        return page, cursor_y
    return _new_pdf_page(pdf_doc)


def _paragraph_font_size(style_name: str) -> float:
    style_name = (style_name or "").lower()
    if "title" in style_name:
        return 18
    if "heading 1" in style_name:
        return 16
    if "heading 2" in style_name:
        return 14
    if "heading 3" in style_name:
        return 13
    return 11


def _paragraph_alignment(paragraph: Paragraph) -> int:
    alignment = paragraph.alignment
    if alignment == 1:
        return 1
    if alignment == 2:
        return 2
    if alignment == 3:
        return 3
    return 0


def _wrap_text_lines(text: str, max_chars: int) -> list[str]:
    wrapped_lines = []
    for raw_line in text.splitlines() or [""]:
        stripped = raw_line.rstrip()
        if not stripped:
            wrapped_lines.append("")
            continue

        current = ""
        for word in stripped.split():
            candidate = word if not current else f"{current} {word}"
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    wrapped_lines.append(current)
                current = word
        if current:
            wrapped_lines.append(current)
    return wrapped_lines or [""]


def _list_prefix(paragraph: Paragraph, counters: defaultdict[int, int]) -> str:
    style_name = (paragraph.style.name or "").lower()
    if "list bullet" in style_name:
        return "• "
    if "list number" in style_name:
        level = 1
        digits = "".join(char for char in style_name if char.isdigit())
        if digits:
            level = int(digits)
        counters[level] += 1
        for key in list(counters.keys()):
            if key > level:
                counters[key] = 0
        return f"{counters[level]}. "
    return ""


def _render_paragraph_to_pdf(
    pdf_doc: fitz.Document, page, cursor_y: float, paragraph: Paragraph, counters
) -> tuple:
    text = paragraph.text or ""
    has_page_break = _paragraph_has_page_break(paragraph)

    if has_page_break and not text.strip():
        return _new_pdf_page(pdf_doc)

    if not text.strip():
        return page, cursor_y + 8

    style_name = paragraph.style.name if paragraph.style else ""
    font_size = _paragraph_font_size(style_name)
    line_height = max(16, font_size * 1.45)
    space_before = _length_to_points(paragraph.paragraph_format.space_before, 0.0)
    space_after = _length_to_points(paragraph.paragraph_format.space_after, 4.0)
    left_indent = _length_to_points(paragraph.paragraph_format.left_indent, 0.0)
    first_indent = _length_to_points(paragraph.paragraph_format.first_line_indent, 0.0)
    prefix = _list_prefix(paragraph, counters)
    rendered_text = prefix + text

    width = page.rect.width - 108 - left_indent
    max_chars = max(24, int(width / max(font_size * 0.55, 1)))
    wrapped_lines = _wrap_text_lines(rendered_text, max_chars)
    estimated_lines = len(wrapped_lines)
    estimated_height = space_before + (estimated_lines * line_height) + space_after
    page, cursor_y = _ensure_page_space(pdf_doc, page, cursor_y, estimated_height)
    cursor_y += space_before

    align = _paragraph_alignment(paragraph)
    for line_index, line in enumerate(wrapped_lines):
        line_indent = left_indent + (max(first_indent, 0) if line_index == 0 else 0)
        x_pos = 54 + line_indent
        if align in (1, 2):
            text_width = _estimate_text_width(line, font_size)
            usable_width = page.rect.width - 108 - line_indent
            if align == 1:
                x_pos = 54 + line_indent + max(0, (usable_width - text_width) / 2)
            else:
                x_pos = 54 + line_indent + max(0, usable_width - text_width)

        page.insert_text(
            fitz.Point(x_pos, cursor_y + font_size),
            line,
            **_font_kwargs(font_size),
        )
        cursor_y += line_height

    cursor_y += space_after
    if has_page_break:
        return _new_pdf_page(pdf_doc)
    return page, cursor_y


def _render_table_to_pdf(
    pdf_doc: fitz.Document, page, cursor_y: float, table: Table
) -> tuple:
    rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
    rows = [row for row in rows if any(cell for cell in row)]
    if not rows:
        return page, cursor_y

    column_count = max(len(row) for row in rows)
    cell_width = (page.rect.width - 108) / max(1, column_count)

    for row in rows:
        max_lines = max(
            (cell.count("\n") + max(1, len(cell) // 28) + 1) for cell in row
        )
        row_height = max(28, max_lines * 14)
        page, cursor_y = _ensure_page_space(pdf_doc, page, cursor_y, row_height + 6)

        for index in range(column_count):
            x0 = 54 + (index * cell_width)
            y0 = cursor_y
            x1 = x0 + cell_width
            y1 = cursor_y + row_height
            page.draw_rect(
                fitz.Rect(x0, y0, x1, y1), color=(0.75, 0.79, 0.89), width=0.8
            )

            text = row[index] if index < len(row) else ""
            page.insert_textbox(
                fitz.Rect(x0 + 6, y0 + 6, x1 - 6, y1 - 6),
                text,
                align=0,
                lineheight=1.25,
                **_font_kwargs(10),
            )

        cursor_y += row_height

    return page, cursor_y + 10


def _convert_docx_with_internal_renderer(
    source_path: Path, destination_pdf: Path
) -> str:
    document = Document(str(source_path))
    pdf_doc = fitz.open()
    page, cursor_y = _new_pdf_page(pdf_doc)
    numbering_counters = defaultdict(int)

    for block in _iter_block_items(document):
        if isinstance(block, Paragraph):
            page, cursor_y = _render_paragraph_to_pdf(
                pdf_doc, page, cursor_y, block, numbering_counters
            )
        else:
            page, cursor_y = _render_table_to_pdf(pdf_doc, page, cursor_y, block)

    pdf_doc.save(destination_pdf)
    pdf_doc.close()
    return "internal_docx_renderer"


def _convert_txt_with_internal_renderer(
    source_path: Path, destination_pdf: Path
) -> str:
    pdf_doc = fitz.open()
    page, cursor_y = _new_pdf_page(pdf_doc)
    font_size = 11
    line_height = 15

    with source_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle.read().splitlines():
            line = raw_line.rstrip("\n")
            if "\f" in line:
                fragments = line.split("\f")
            else:
                fragments = [line]

            for fragment_index, fragment in enumerate(fragments):
                page, cursor_y = _ensure_page_space(
                    pdf_doc, page, cursor_y, line_height + 6
                )
                page.insert_text(
                    fitz.Point(54, cursor_y),
                    fragment,
                    **_font_kwargs(font_size),
                )
                cursor_y += line_height
                if fragment_index < len(fragments) - 1:
                    page, cursor_y = _new_pdf_page(pdf_doc)

    pdf_doc.save(destination_pdf)
    pdf_doc.close()
    return "internal_txt_renderer"


def _convert_via_gotenberg(source_path: Path, destination_pdf: Path) -> None:
    import urllib.error
    import urllib.request
    from uuid import uuid4

    boundary = uuid4().hex
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    filename = source_path.name

    with open(source_path, "rb") as f:
        file_content = f.read()

    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="files"; filename="{filename}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        + file_content
        + f"\r\n--{boundary}--\r\n".encode()
    )

    req = urllib.request.Request(
        "http://localhost:3000/forms/libreoffice/convert",
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            with open(destination_pdf, "wb") as f:
                f.write(response.read())
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Gotenberg API failed. Please ensure Gotenberg docker is running on port 3000. Detail: {e}"
        )


def convert_to_pdf(source_path: str | Path, destination_pdf: str | Path) -> str:
    source_path = Path(source_path)
    destination_pdf = Path(destination_pdf)
    suffix = source_path.suffix.lower()

    if suffix == ".pdf":
        return _copy_to_canonical_pdf(source_path, destination_pdf)

    if suffix not in SUPPORTED_INPUT_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {suffix}")

    if destination_pdf.exists():
        try:
            destination_pdf.unlink()
        except OSError:
            pass

    if suffix == ".txt":
        return _convert_txt_with_internal_renderer(source_path, destination_pdf)

    try:
        _convert_via_gotenberg(source_path, destination_pdf)
        return "gotenberg"
    except Exception as e:
        pass

    if suffix == ".docx":
        if os.name == "nt":
            try:
                _run_word_export_via_powershell(source_path, destination_pdf)
                return "ms_word_com"
            except Exception:
                pass

        try:
            _run_libreoffice_export(source_path, destination_pdf)
            return "libreoffice"
        except Exception:
            return _convert_docx_with_internal_renderer(source_path, destination_pdf)

    if suffix == ".txt":
        return _convert_txt_with_internal_renderer(source_path, destination_pdf)

    raise ValueError(f"Cannot convert file to PDF: {source_path}")


def _extract_text_and_analyse_pdf(pdf_path: Path) -> dict[str, Any]:
    pages = []
    total_chars = 0
    low_text_pages = 0
    image_heavy_pages = 0
    records = []
    current_char = 0

    with fitz.open(pdf_path) as pdf_doc:
        for page_index, page in enumerate(pdf_doc, start=1):
            page_dict = page.get_text("dict")
            line_texts = []
            page_char_count = 0

            for block_index, block in enumerate(page_dict.get("blocks", [])):
                if block.get("type") != 0:
                    continue

                for line_index, line in enumerate(block.get("lines", [])):
                    spans = line.get("spans", [])
                    text = "".join(span.get("text", "") for span in spans)
                    normalized_text = _normalize_pdf_line_text(text)
                    if not normalized_text:
                        continue

                    bbox = [round(value, 2) for value in line.get("bbox", [])]
                    text_items = []
                    for span in spans:
                        span_text = _normalize_pdf_line_text(span.get("text", ""))
                        if not span_text:
                            continue
                        span_bbox = [round(value, 2) for value in span.get("bbox", [])]
                        text_items.append(
                            {
                                "text": span_text,
                                "bbox": span_bbox,
                            }
                        )

                    char_start = current_char
                    char_end = current_char + len(normalized_text)
                    current_char = char_end + 1

                    record = {
                        "content": normalized_text,
                        "canonical": normalized_text,
                        "retrieval": normalized_text.lower(),
                        "char_start": char_start,
                        "char_end": char_end,
                        "page_number": page_index,
                        "source_file": str(pdf_path),
                        "bbox": bbox,
                        "text_items": text_items,
                        "block_index": block_index,
                        "line_index": line_index,
                    }
                    records.append(record)
                    line_texts.append(normalized_text)
                    page_char_count += len(normalized_text)

            page_text = "\n".join(line_texts).strip()
            page_images = len(page.get_images(full=True))
            total_chars += page_char_count

            if page_char_count < 80:
                low_text_pages += 1
            if page_images > 0:
                image_heavy_pages += 1

            pages.append(page_text)

    page_count = max(1, len(pages))
    combined_text = "\n".join(page for page in pages if page).strip()
    average_chars = total_chars / page_count
    scan_based = low_text_pages / page_count >= 0.6 and image_heavy_pages > 0
    weak_extraction = total_chars < max(160, page_count * 60)
    can_ingest = total_chars > 0

    warning = None
    if scan_based and weak_extraction:
        warning = (
            "PDF looks scan/image-based and extracted text is very limited. "
            "OCR is not implemented yet, so ingest/compare may be unreliable."
        )
    elif weak_extraction:
        warning = "PDF extraction returned very little text. Please provide a PDF with a text layer for reliable ingest."

    return {
        "text": combined_text,
        "records": records,
        "page_count": page_count,
        "total_chars": total_chars,
        "average_chars_per_page": average_chars,
        "scan_based": scan_based,
        "weak_extraction": weak_extraction,
        "can_ingest": can_ingest,
        "warning": warning,
    }


def prepare_document(
    file_path: str,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    cache_key: str | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    if progress_callback:
        progress_callback(5, "Đang kiểm tra file đầu vào")

    source_path = Path(file_path).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"File not found: {source_path}")

    suffix = source_path.suffix.lower()
    if suffix not in SUPPORTED_INPUT_EXTENSIONS:
        raise ValueError(f"Unsupported input format: {suffix}")

    cache_root = Path(cache_root)
    directories = _ensure_cache_dirs(cache_root)
    source_hash = compute_file_hash(source_path)
    cache_key = cache_key or source_hash
    normalized_pdf_path = directories["normalized"] / f"{cache_key}.pdf"
    text_cache_path = directories["text"] / f"{cache_key}.txt"
    records_cache_path = directories["records"] / f"{cache_key}.json"

    if progress_callback:
        progress_callback(20, "Đang chuẩn hóa PDF")

    conversion_method = None
    if suffix == ".pdf":
        conversion_method = _copy_to_canonical_pdf(source_path, normalized_pdf_path)
    else:
        conversion_method = convert_to_pdf(source_path, normalized_pdf_path)

    if progress_callback:
        progress_callback(45, "Đang trích xuất text từ PDF")
    analysis = _extract_text_and_analyse_pdf(normalized_pdf_path)
    extracted_text = analysis["text"]
    extracted_records = analysis["records"]
    text_cache_path.write_text(extracted_text, encoding="utf-8")
    _json_dump(records_cache_path, extracted_records)

    if progress_callback:
        progress_callback(70, "Đang hoàn tất metadata tài liệu")

    prepared = {
        "source_path": str(source_path),
        "source_hash": source_hash,
        "cache_key": cache_key,
        "original_extension": suffix,
        "normalized_pdf_path": str(normalized_pdf_path.resolve()),
        "preview_pdf_path": str(normalized_pdf_path.resolve()),
        "conversion_method": conversion_method,
        "text_cache_path": str(text_cache_path.resolve()),
        "records_cache_path": str(records_cache_path.resolve()),
        "text": extracted_text,
        "records": extracted_records,
        "page_count": analysis["page_count"],
        "total_chars": analysis["total_chars"],
        "scan_based": analysis["scan_based"],
        "weak_extraction": analysis["weak_extraction"],
        "can_ingest": analysis["can_ingest"],
        "warning": analysis["warning"],
        "prepared_at": _utc_now(),
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
    }

    return prepared


def build_chunk_metadata(
    chunk: dict[str, Any], doc_id: str, clause_index: int, prepared_doc: dict[str, Any]
) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "article": chunk.get("article", ""),
        "clause_index": clause_index,
        "source_file": prepared_doc["normalized_pdf_path"],
        "preview_pdf_path": prepared_doc["preview_pdf_path"],
        "source_hash": prepared_doc["source_hash"],
        "conversion_method": prepared_doc["conversion_method"],
        "scan_based": prepared_doc["scan_based"],
        "warning": prepared_doc["warning"] or "",
        "page_number": chunk.get("page_number"),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
    }


def ingest_document(
    file_path: str,
    doc_id: str,
    db_path: str = "legal_data.json",
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    cache_key: str | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    cache_root = Path(cache_root)
    directories = _ensure_cache_dirs(cache_root)
    prepared = prepare_document(
        file_path=file_path,
        cache_root=cache_root,
        cache_key=cache_key or doc_id,
        progress_callback=(lambda p, m: progress_callback(min(75, p), m))
        if progress_callback
        else None,
    )
    db = LegalVectorDB(db_name=db_path)

    if progress_callback:
        progress_callback(78, "Đang kiểm tra cache ingest")
    vector_cache_path = directories["vectors"] / f"{prepared['cache_key']}.json"
    chunk_cache_path = directories["chunks"] / f"{prepared['cache_key']}.json"

    if not prepared["can_ingest"]:
        if progress_callback:
            progress_callback(90, "Text quá ít, bỏ qua ingest và trả cảnh báo")
        result = {
            **prepared,
            "doc_id": doc_id,
            "ingest_status": "warning_not_ingested",
            "used_vector_cache": False,
            "chunks_count": 0,
            "vector_cache_path": str(vector_cache_path.resolve()),
            "chunk_cache_path": str(chunk_cache_path.resolve()),
        }
        if progress_callback:
            progress_callback(100, "Hoàn tất với cảnh báo")
        return result

    if progress_callback:
        progress_callback(86, "Đang chunking văn bản")

    chunks = structural_chunking(
        prepared["text"], doc_id=doc_id, records=prepared.get("records")
    )
    metadata_list = [
        build_chunk_metadata(chunk, doc_id, index, prepared)
        for index, chunk in enumerate(chunks)
    ]
    if progress_callback:
        progress_callback(92, "Đang tạo embedding và ingest VectorDB")

    inserted_entries = db.insert_legal_chunks(
        chunks, metadata_list, doc_id_to_clear=doc_id
    )

    _json_dump(vector_cache_path, inserted_entries)
    _json_dump(chunk_cache_path, chunks)

    ingest_status = "completed_with_warning" if prepared["warning"] else "completed"
    result = {
        **prepared,
        "doc_id": doc_id,
        "ingest_status": ingest_status,
        "used_vector_cache": False,
        "chunks_count": len(chunks),
        "vector_cache_path": str(vector_cache_path.resolve()),
        "chunk_cache_path": str(chunk_cache_path.resolve()),
    }

    if progress_callback:
        progress_callback(100, "Ingest hoàn tất")

    return result


def extract_text(file_path: str) -> str:
    prepared = prepare_document(file_path)
    return prepared["text"]
