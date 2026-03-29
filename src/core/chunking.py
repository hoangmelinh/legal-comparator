import re
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple


_RE_PART    = re.compile(r"^(PHẦN|CHƯƠNG|PART|CHAPTER)\s+([IVXLCDM\d]+)\b(.*)", re.IGNORECASE)
_RE_SECTION = re.compile(r"^(MỤC|SECTION)\s+(\d+|[IVXLCDM]+)\b(.*)",         re.IGNORECASE)


_RE_ARTICLE = re.compile(
    r"^(?:Điều|Clause|Article)\s+(\d+)\s*[.:\-–]?\s*(.*)",
    re.IGNORECASE,
)


_RE_CLAUSE = re.compile(r"^(\d+)\.\s+(.+)", re.DOTALL)


_RE_POINT = re.compile(r"^([a-zđ]{1,2})\)\s+(.+)", re.DOTALL)


_RE_BULLET = re.compile(r"^[-•+]\s+(.+)", re.DOTALL)


_STRUCTURAL_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("part",    _RE_PART),
    ("section", _RE_SECTION),
    ("article", _RE_ARTICLE),
    ("clause",  _RE_CLAUSE),
    ("point",   _RE_POINT),
    ("bullet",  _RE_BULLET),
]

@dataclass
class LegalChunk:
    chunk_id      : str
    parent_law_id : str                   # "Điều 5"
    clause_id     : Optional[int]         # 2 (khoản 2), hoặc None
    point_id      : Optional[str]         # "a", "b", hoặc None
    context       : str                   # Tiêu đề Điều cha
    breadcrumb    : str                   # "Chương I > Điều 5 > Khoản 2"
    content       : str                   # Nội dung (canonical)
    retrieval_text: str                   # Nội dung đã chuẩn hóa
    source_file   : str
    page_number   : Optional[int]
    char_start    : int
    char_end      : int
    level         : int                   # 2=Điều, 3=Khoản, 4=Điểm
    sub_points    : List[str] = field(default_factory=list)
    token_count   : int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class _ArticleBuffer:
    """Buffer nội bộ khi đang tích lũy nội dung của một Điều."""
    article_num  : int
    article_title: str
    page_number  : Optional[int]
    char_start   : int
    source_file  : str
    breadcrumb   : str                              # context bên ngoài Điều
    lines        : List[Dict[str, Any]] = field(default_factory=list)

def _classify_line(text: str) -> Tuple[str, re.Match]:
    stripped = text.strip()
    for line_type, pattern in _STRUCTURAL_PATTERNS:
        m = pattern.match(stripped)
        if m:
            return line_type, m
    return "body", None

def _estimate_tokens(text: str) -> int:
    words = len(text.split())
    return max(1, int(words / 0.75))

def legal_chunker(
    records: List[Dict[str, Any]],
    normalize: bool = True,
    merge_body_to_clause: bool = True,
    min_chunk_chars: int = 10,
) -> List[LegalChunk]:
    chunks: List[LegalChunk] = []
    
    current_part    : str = ""
    current_section : str = ""
    current_article : Optional[_ArticleBuffer] = None

    seen_article_labels: Dict[str, int] = {}

    def _get_text(record: Dict) -> str:
        return record.get("canonical") or record.get("content", "")

    def _get_retrieval(record: Dict) -> str:
        if normalize:
            return record.get("retrieval") or _get_text(record)
        return _get_text(record)

    def _build_breadcrumb(article_num: Optional[int] = None,
                          clause_num: Optional[int] = None,
                          point_str: Optional[str] = None) -> str:
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

    def _flush_article_buffer(buf: _ArticleBuffer) -> List[LegalChunk]:
        result: List[LegalChunk] = []
        
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

        groups: List[Dict[str, Any]] = []
        current_clause: Optional[Dict] = None
        current_point : Optional[Dict] = None
        body_before_clause: List[str] = []

        for line_info in buf.lines:
            lt   = line_info["line_type"]
            text = line_info["text"]

            if lt == "clause":
                if current_point and current_clause:
                    current_clause["points"].append(current_point)
                    current_point = None
                if current_clause:
                    groups.append(current_clause)

                m = _RE_CLAUSE.match(text)
                clause_num  = int(m.group(1)) if m else None
                clause_body = m.group(2) if m else text

                current_clause = {
                    "clause_num"   : clause_num,
                    "clause_text"  : clause_body,
                    "points"       : [],
                    "body_lines"   : [],
                    "char_start"   : line_info["char_start"],
                    "char_end"     : line_info["char_end"],
                    "page_number"  : line_info.get("page"),
                }

            elif lt == "point":
                if current_point and current_clause:
                    current_clause["points"].append(current_point)

                m = _RE_POINT.match(text)
                point_id   = m.group(1) if m else None
                point_body = m.group(2) if m else text

                current_point = {
                    "point_id"  : point_id,
                    "point_text": point_body,
                    "bullets"   : [],
                    "char_start": line_info["char_start"],
                    "char_end"  : line_info["char_end"],
                    "page_number": line_info.get("page"),
                }

            elif lt == "bullet":
                m   = _RE_BULLET.match(text)
                txt = m.group(1) if m else text
                if current_point:
                    current_point["bullets"].append(txt)
                elif current_clause:
                    current_clause["body_lines"].append(txt)

            else:  # body
                if current_clause:
                    current_clause["body_lines"].append(text)
                elif merge_body_to_clause:
                    body_before_clause.append(text)

        if current_point and current_clause:
            current_clause["points"].append(current_point)
        if current_clause:
            groups.append(current_clause)

        if not groups:
            all_body = " ".join(body_before_clause).strip()
            content  = f"{article_context}\n{all_body}".strip() if all_body else article_context
            if len(content) >= min_chunk_chars:
                c_start = buf.char_start
                c_end   = buf.char_start + len(content)
                result.append(LegalChunk(
                    chunk_id       = str(uuid.uuid4()),
                    parent_law_id  = article_label,
                    clause_id      = None,
                    point_id       = None,
                    context        = article_context,
                    breadcrumb     = _build_breadcrumb(buf.article_num),
                    content        = content,
                    retrieval_text = content.lower(),
                    source_file    = buf.source_file,
                    page_number    = buf.page_number,
                    char_start     = c_start,
                    char_end       = c_end,
                    level          = 2,
                    sub_points     = [],
                    token_count    = _estimate_tokens(content),
                ))
            return result

        for grp in groups:
            clause_num = grp["clause_num"]
            clause_lines = [grp["clause_text"]] + grp["body_lines"]

            sub_point_texts = []
            for pt in grp["points"]:
                pt_text = f"{pt['point_id']}) {pt['point_text']}"
                if pt["bullets"]:
                    pt_text += "\n" + "\n".join(f"  - {b}" for b in pt["bullets"])
                clause_lines.append(pt_text)
                sub_point_texts.append(pt_text)

            raw_content = "\n".join(line for line in clause_lines if line.strip())
            context_prefix = f"[{article_context}]"
            full_content   = f"{context_prefix}\n{raw_content}".strip()

            if len(full_content) < min_chunk_chars:
                continue

            result.append(LegalChunk(
                chunk_id       = str(uuid.uuid4()),
                parent_law_id  = article_label,
                clause_id      = clause_num,
                point_id       = None,
                context        = article_context,
                breadcrumb     = _build_breadcrumb(buf.article_num, clause_num),
                content        = full_content,
                retrieval_text = full_content.lower(),
                source_file    = buf.source_file,
                page_number    = grp.get("page_number") or buf.page_number,
                char_start     = grp["char_start"],
                char_end       = grp["char_end"],
                level          = 3,
                sub_points     = sub_point_texts,
                token_count    = _estimate_tokens(full_content),
            ))

        return result

    for record in records:
        text        = _get_text(record)
        retrieval   = _get_retrieval(record)
        char_start  = record.get("char_start", 0)
        char_end    = record.get("char_end",   char_start + len(text))
        page_number = record.get("page_number")
        source_file = record.get("source_file", "")

        if not text.strip():
            continue

        line_type, match = _classify_line(text)

        if line_type == "part":
            if current_article:
                chunks.extend(_flush_article_buffer(current_article))
                current_article = None
            current_part    = f"{match.group(1).capitalize()} {match.group(2)}"
            current_section = ""

        elif line_type == "section":
            if current_article:
                chunks.extend(_flush_article_buffer(current_article))
                current_article = None
            current_section = f"Mục {match.group(1)}"

        elif line_type == "article":
            if current_article:
                chunks.extend(_flush_article_buffer(current_article))

            article_num   = int(match.group(1))
            article_title = match.group(2).strip().rstrip(".").strip()

            current_article = _ArticleBuffer(
                article_num   = article_num,
                article_title = article_title,
                page_number   = page_number,
                char_start    = char_start,
                source_file   = source_file,
                breadcrumb    = _build_breadcrumb(),
            )

        elif line_type in ("clause", "point", "bullet", "body"):
            if current_article is None:
                current_article = _ArticleBuffer(
                    article_num   = 0,
                    article_title = "Nội dung không phân điều",
                    page_number   = page_number,
                    char_start    = char_start,
                    source_file   = source_file,
                    breadcrumb    = "Tổng quan",
                )
            
            current_article.lines.append({
                "text"      : text,
                "line_type" : line_type,
                "char_start": char_start,
                "char_end"  : char_end,
                "page"      : page_number,
            })

    if current_article:
        chunks.extend(_flush_article_buffer(current_article))

    return chunks

def structural_chunking(text: str, doc_id: str = "doc1") -> List[Dict[str, Any]]:
    """
    Hàm giao tiếp với pipeline hiện tại (main.py).
    Chuyển văn bản thô thành danh sách records rồi đưa vào legal_chunker.
    """
    # Nếu văn bản hoàn toàn không có cấu trúc Điều khoản pháp lý, ta dùng chế độ Fallback (chia theo đoạn)
    if not re.search(r"^(?:Điều|Clause|Article)\s+(\d+)", text, re.IGNORECASE | re.MULTILINE):
        lines = text.split("\n")
        final_chunks = []
        current_chunk_lines = []
        current_word_count = 0
        chunk_index = 1
        
        for line in lines:
            line = line.strip()
            if not line: continue
            current_chunk_lines.append(line)
            current_word_count += len(line.split())
            
            # Ngắt chunk nếu đủ dài (>=100 từ) hoặc có dấu hiệu ngắt câu
            if current_word_count >= 100:
                para_text = "\n".join(current_chunk_lines)
                final_chunks.append({
                    "doc_id": doc_id,
                    "article": f"Đoạn {chunk_index}",
                    "content": para_text,
                })
                chunk_index += 1
                current_chunk_lines = []
                current_word_count = 0
                
        if current_chunk_lines:
            para_text = "\n".join(current_chunk_lines)
            final_chunks.append({
                "doc_id": doc_id,
                "article": f"Đoạn {chunk_index}",
                "content": para_text,
            })
        return final_chunks

    # Tách văn bản thành từng dòng (Chế độ chuẩn có cấu trúc)
    lines = text.split("\n")
    records = []
    
    current_char = 0
    for line in lines:
        line_len = len(line)
        records.append({
            "content": line,
            "char_start": current_char,
            "char_end": current_char + line_len,
            "source_file": doc_id,
        })
        current_char += line_len + 1  # +1 cho ký tự newline
        
    # Gọi chunker mới
    legal_chunks = legal_chunker(records, normalize=False)
    
    # Chuyển đổi về list dict như cấu trúc cũ đôi chút (tương thích db insert).
    # vector_store.py chấp nhận list dict, nó sẽ lấy item["content"]
    final_chunks = []
    for chunk in legal_chunks:
        chunk_dict = chunk.to_dict()
        # Bổ sung key doc_id và article để tương thích với luồng cũ
        chunk_dict["doc_id"] = doc_id
        chunk_dict["article"] = chunk.parent_law_id
        final_chunks.append(chunk_dict)
        
    return final_chunks