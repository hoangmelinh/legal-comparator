import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

_RE_PART = re.compile(
    r"^(PHẦN|CHƯƠNG|PART|CHAPTER)\s+([IVXLCDM\d]+)\b(.*)", re.IGNORECASE
)
_RE_SECTION = re.compile(r"^(MỤC|SECTION)\s+(\d+|[IVXLCDM]+)\b(.*)", re.IGNORECASE)
_RE_ARTICLE = re.compile(
    r"^(?:Điều|Clause|Article)\s+(\d+)\s*[.:\-–]?\s*(.*)", re.IGNORECASE
)
_RE_CLAUSE = re.compile(r"^(\d+)\.\s+(.+)", re.DOTALL)
_RE_POINT = re.compile(r"^([a-zđ]{1,2})\)\s+(.+)", re.DOTALL)
_RE_BULLET = re.compile(r"^[-•+]\s+(.+)", re.DOTALL)

_STRUCTURAL_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("part", _RE_PART),
    ("section", _RE_SECTION),
    ("article", _RE_ARTICLE),
    ("clause", _RE_CLAUSE),
    ("point", _RE_POINT),
    ("bullet", _RE_BULLET),
]


@dataclass
class LegalChunk:
    chunk_id: str
    parent_law_id: str
    clause_id: int | None
    point_id: str | None
    context: str
    breadcrumb: str
    content: str
    retrieval_text: str
    source_file: str
    page_number: int | None
    char_start: int
    char_end: int
    level: int
    sub_points: list[str] = field(default_factory=list)
    token_count: int = 0
    page_start: int | None = None
    page_end: int | None = None
    source_anchors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _ArticleBuffer:
    article_num: int
    article_title: str
    page_number: int | None
    char_start: int
    source_file: str
    breadcrumb: str
    lines: list[dict[str, Any]] = field(default_factory=list)


def _classify_line(text: str) -> tuple[str, re.Match | None]:
    stripped = text.strip()
    for line_type, pattern in _STRUCTURAL_PATTERNS:
        match = pattern.match(stripped)
        if match:
            return line_type, match
    return "body", None


def _estimate_tokens(text: str) -> int:
    words = len(text.split())
    return max(1, int(words / 0.75))


def _compact_anchor(record: dict[str, Any]) -> dict[str, Any]:
    anchor = {
        "page": record.get("page")
        if record.get("page") is not None
        else record.get("page_number"),
        "text": record.get("text")
        or record.get("content")
        or record.get("canonical")
        or "",
        "char_start": record.get("char_start"),
        "char_end": record.get("char_end"),
    }
    if record.get("bbox") is not None:
        anchor["bbox"] = record.get("bbox")
    if record.get("text_items") is not None:
        anchor["text_items"] = record.get("text_items")
    if record.get("block_index") is not None:
        anchor["block_index"] = record.get("block_index")
    if record.get("line_index") is not None:
        anchor["line_index"] = record.get("line_index")
    return anchor


def _page_span(anchors: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    pages = [anchor.get("page") for anchor in anchors if anchor.get("page") is not None]
    if not pages:
        return None, None
    return min(pages), max(pages)


def legal_chunker(
    records: list[dict[str, Any]],
    normalize: bool = True,
    merge_body_to_clause: bool = True,
    min_chunk_chars: int = 10,
) -> list[LegalChunk]:
    chunks: list[LegalChunk] = []
    current_part = ""
    current_section = ""
    current_article: _ArticleBuffer | None = None
    seen_article_labels: dict[str, int] = {}

    def _get_text(record: dict[str, Any]) -> str:
        return record.get("canonical") or record.get("content", "")

    def _get_retrieval(record: dict[str, Any]) -> str:
        if normalize:
            return record.get("retrieval") or _get_text(record)
        return _get_text(record)

    def _build_breadcrumb(
        article_num: int | None = None,
        clause_num: int | None = None,
        point_str: str | None = None,
    ) -> str:
        parts = []
        if current_part:
            parts.append(current_part)
        if current_section:
            parts.append(current_section)
        if article_num is not None:
            parts.append(f"Điều {article_num}")
        if clause_num is not None:
            parts.append(f"Khoản {clause_num}")
        if point_str is not None:
            parts.append(f"Điểm {point_str}")
        return " > ".join(parts) if parts else "Tổng quan"

    def _flush_article_buffer(buf: _ArticleBuffer) -> list[LegalChunk]:
        result: list[LegalChunk] = []

        base_label = _build_breadcrumb(buf.article_num)
        seen_article_labels[base_label] = seen_article_labels.get(base_label, 0) + 1
        if seen_article_labels[base_label] > 1:
            article_label = f"{base_label} (Lần {seen_article_labels[base_label]})"
        else:
            article_label = base_label

        article_context = (
            f"{article_label}. {buf.article_title}"
            if buf.article_title
            else article_label
        )

        groups: list[dict[str, Any]] = []
        current_clause: dict[str, Any] | None = None
        current_point: dict[str, Any] | None = None
        body_before_clause: list[dict[str, Any]] = []

        for line_info in buf.lines:
            line_type = line_info["line_type"]
            text = line_info["text"]

            if line_type == "clause":
                if current_point and current_clause:
                    current_clause["points"].append(current_point)
                    current_point = None
                if current_clause:
                    groups.append(current_clause)

                match = _RE_CLAUSE.match(text)
                clause_num = int(match.group(1)) if match else None
                clause_body = match.group(2) if match else text
                current_clause = {
                    "clause_num": clause_num,
                    "clause_text": clause_body,
                    "points": [],
                    "body_lines": [],
                    "anchors": [_compact_anchor(line_info)],
                    "char_start": line_info["char_start"],
                    "char_end": line_info["char_end"],
                    "page_number": line_info.get("page"),
                }
                continue

            if line_type == "point":
                if current_point and current_clause:
                    current_clause["points"].append(current_point)

                match = _RE_POINT.match(text)
                point_id = match.group(1) if match else None
                point_body = match.group(2) if match else text
                current_point = {
                    "point_id": point_id,
                    "point_text": point_body,
                    "bullets": [],
                    "anchors": [_compact_anchor(line_info)],
                    "char_start": line_info["char_start"],
                    "char_end": line_info["char_end"],
                    "page_number": line_info.get("page"),
                }
                continue

            if line_type == "bullet":
                match = _RE_BULLET.match(text)
                bullet_text = match.group(1) if match else text
                if current_point:
                    current_point["bullets"].append(bullet_text)
                    current_point["anchors"].append(_compact_anchor(line_info))
                elif current_clause:
                    current_clause["body_lines"].append(bullet_text)
                    current_clause["anchors"].append(_compact_anchor(line_info))
                continue

            if current_clause:
                current_clause["body_lines"].append(text)
                current_clause["anchors"].append(_compact_anchor(line_info))
            elif merge_body_to_clause:
                body_before_clause.append(line_info)

        if current_point and current_clause:
            current_clause["points"].append(current_point)
        if current_clause:
            groups.append(current_clause)

        if not groups:
            article_anchors = [_compact_anchor(line) for line in body_before_clause]
            page_start, page_end = _page_span(article_anchors)
            all_body = " ".join(
                line.get("text", "") for line in body_before_clause
            ).strip()
            content = (
                f"{article_context}\n{all_body}".strip()
                if all_body
                else article_context
            )

            if len(content) >= min_chunk_chars:
                result.append(
                    LegalChunk(
                        chunk_id=str(uuid.uuid4()),
                        parent_law_id=article_label,
                        clause_id=None,
                        point_id=None,
                        context=article_context,
                        breadcrumb=_build_breadcrumb(buf.article_num),
                        content=content,
                        retrieval_text=content.lower(),
                        source_file=buf.source_file,
                        page_number=buf.page_number,
                        char_start=buf.char_start,
                        char_end=buf.char_start + len(content),
                        level=2,
                        sub_points=[],
                        token_count=_estimate_tokens(content),
                        page_start=page_start,
                        page_end=page_end,
                        source_anchors=article_anchors,
                    )
                )
            return result

        for group in groups:
            clause_lines = [group["clause_text"]] + group["body_lines"]
            sub_point_texts = []

            for point in group["points"]:
                point_text = f"{point['point_id']}) {point['point_text']}"
                if point["bullets"]:
                    point_text += "\n" + "\n".join(
                        f"  - {bullet}" for bullet in point["bullets"]
                    )
                clause_lines.append(point_text)
                sub_point_texts.append(point_text)
                group["anchors"].extend(point.get("anchors", []))

            raw_content = "\n".join(line for line in clause_lines if line.strip())
            full_content = f"[{article_context}]\n{raw_content}".strip()
            if len(full_content) < min_chunk_chars:
                continue

            page_start, page_end = _page_span(group["anchors"])
            result.append(
                LegalChunk(
                    chunk_id=str(uuid.uuid4()),
                    parent_law_id=article_label,
                    clause_id=group["clause_num"],
                    point_id=None,
                    context=article_context,
                    breadcrumb=_build_breadcrumb(buf.article_num, group["clause_num"]),
                    content=full_content,
                    retrieval_text=full_content.lower(),
                    source_file=buf.source_file,
                    page_number=group.get("page_number") or buf.page_number,
                    char_start=group["char_start"],
                    char_end=group["char_end"],
                    level=3,
                    sub_points=sub_point_texts,
                    token_count=_estimate_tokens(full_content),
                    page_start=page_start,
                    page_end=page_end,
                    source_anchors=group["anchors"],
                )
            )

        return result

    for record in records:
        text = _get_text(record)
        _ = _get_retrieval(record)
        char_start = record.get("char_start", 0)
        char_end = record.get("char_end", char_start + len(text))
        page_number = record.get("page_number")
        source_file = record.get("source_file", "")

        if not text.strip():
            continue

        line_type, match = _classify_line(text)
        if line_type == "part":
            if current_article:
                chunks.extend(_flush_article_buffer(current_article))
                current_article = None
            current_part = f"{match.group(1).capitalize()} {match.group(2)}"
            current_section = ""
            continue

        if line_type == "section":
            if current_article:
                chunks.extend(_flush_article_buffer(current_article))
                current_article = None
            current_section = f"Mục {match.group(1)}"
            continue

        if line_type == "article":
            if current_article:
                chunks.extend(_flush_article_buffer(current_article))

            current_article = _ArticleBuffer(
                article_num=int(match.group(1)),
                article_title=match.group(2).strip().rstrip(".").strip(),
                page_number=page_number,
                char_start=char_start,
                source_file=source_file,
                breadcrumb=_build_breadcrumb(),
            )
            continue

        if current_article is None:
            current_article = _ArticleBuffer(
                article_num=0,
                article_title="Nội dung không phân điều",
                page_number=page_number,
                char_start=char_start,
                source_file=source_file,
                breadcrumb="Tổng quan",
            )

        current_article.lines.append(
            {
                "text": text,
                "line_type": line_type,
                "char_start": char_start,
                "char_end": char_end,
                "page": page_number,
                "bbox": record.get("bbox"),
                "text_items": record.get("text_items"),
                "block_index": record.get("block_index"),
                "line_index": record.get("line_index"),
            }
        )

    if current_article:
        chunks.extend(_flush_article_buffer(current_article))

    return chunks


def structural_chunking(
    text: str, doc_id: str = "doc1", records: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    if not re.search(
        r"^(?:Điều|Clause|Article)\s+(\d+)", text, re.IGNORECASE | re.MULTILINE
    ):
        line_records = (
            records
            if records is not None
            else [{"content": line} for line in text.split("\n")]
        )
        final_chunks = []
        current_chunk_lines = []
        current_chunk_anchors = []
        current_word_count = 0
        chunk_index = 1

        for record in line_records:
            line = (record.get("content") or record.get("canonical") or "").strip()
            if not line:
                continue

            current_chunk_lines.append(line)
            current_chunk_anchors.append(_compact_anchor(record))
            current_word_count += len(line.split())

            if current_word_count >= 100:
                para_text = "\n".join(current_chunk_lines)
                page_start, page_end = _page_span(current_chunk_anchors)
                final_chunks.append(
                    {
                        "doc_id": doc_id,
                        "article": f"Đoạn {chunk_index}",
                        "content": para_text,
                        "page_start": page_start,
                        "page_end": page_end,
                        "source_anchors": current_chunk_anchors,
                    }
                )
                chunk_index += 1
                current_chunk_lines = []
                current_chunk_anchors = []
                current_word_count = 0

        if current_chunk_lines:
            para_text = "\n".join(current_chunk_lines)
            page_start, page_end = _page_span(current_chunk_anchors)
            final_chunks.append(
                {
                    "doc_id": doc_id,
                    "article": f"Đoạn {chunk_index}",
                    "content": para_text,
                    "page_start": page_start,
                    "page_end": page_end,
                    "source_anchors": current_chunk_anchors,
                }
            )
        return final_chunks

    if records is None:
        records = []
        current_char = 0
        for line in text.split("\n"):
            line_len = len(line)
            records.append(
                {
                    "content": line,
                    "char_start": current_char,
                    "char_end": current_char + line_len,
                    "source_file": doc_id,
                }
            )
            current_char += line_len + 1

    legal_chunks = legal_chunker(records, normalize=False)
    final_chunks = []
    for chunk in legal_chunks:
        chunk_dict = chunk.to_dict()
        chunk_dict["doc_id"] = doc_id
        chunk_dict["article"] = chunk.parent_law_id
        final_chunks.append(chunk_dict)
    return final_chunks
