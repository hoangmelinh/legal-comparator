"""
Comparator pipeline for clause-by-clause legal document comparison.

The implementation keeps the existing project spirit:
- vector DB is still the source for clause retrieval
- Ollama is still used for substantive MODIFIED analysis
- lexical / semantic heuristics are only used to bypass obvious identical
  or minor-wording changes and to refine highlight spans
"""

from __future__ import annotations

import concurrent.futures
import difflib
import json
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Callable, Dict, List, Optional

import numpy as np

from src.core.llm import DEFAULT_MODEL, compare_clauses
from src.database.vector_store import LegalVectorDB


_TOKEN_REGEX = re.compile(r"\d+(?:[.,]\d+)*%?|\w+|[^\w\s]", re.UNICODE)
_NUMERIC_REGEX = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_DATE_REGEX = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
_TIME_REGEX = re.compile(r"\b\d{1,2}:\d{2}(?:\s?(?:AM|PM|am|pm))?\b")


def _comparison_tokens(text: str) -> list[dict]:
    return [
        token
        for token in _tokenize_with_spans(text)
        if re.search(r"\w|\d", token.get("value", ""))
    ]


def _extract_numeric_signals(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text or "")
    signals: list[str] = []
    signals.extend(_NUMERIC_REGEX.findall(normalized))
    signals.extend(_DATE_REGEX.findall(normalized))
    signals.extend(_TIME_REGEX.findall(normalized))

    tokens = _comparison_tokens(normalized)
    for index, token in enumerate(tokens):
        value = token.get("value", "")
        if not re.search(r"\d", value):
            continue

        token_parts = [token["norm"]]
        if index + 1 < len(tokens):
            next_value = tokens[index + 1].get("value", "")
            if re.fullmatch(r"[^\W\d_]+", next_value, re.UNICODE):
                token_parts.append(tokens[index + 1]["norm"])
        signals.append(" ".join(token_parts).strip())

    return sorted({signal.strip().lower() for signal in signals if signal})


def _token_change_stats(text_a: str, text_b: str) -> dict:
    tokens_a = _comparison_tokens(text_a)
    tokens_b = _comparison_tokens(text_b)
    norms_a = [token["norm"] for token in tokens_a]
    norms_b = [token["norm"] for token in tokens_b]
    matcher = SequenceMatcher(None, norms_a, norms_b)

    changed_a = 0
    changed_b = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            changed_a += max(0, i2 - i1)
        if tag in ("replace", "insert"):
            changed_b += max(0, j2 - j1)

    return {
        "tokens_a": tokens_a,
        "tokens_b": tokens_b,
        "norms_a": norms_a,
        "norms_b": norms_b,
        "ratio": matcher.ratio(),
        "changed_a": changed_a,
        "changed_b": changed_b,
    }


def _tokenize_with_spans(text: str) -> list[dict]:
    tokens = []
    for match in _TOKEN_REGEX.finditer(text or ""):
        value = match.group(0)
        if not value.strip():
            continue
        tokens.append(
            {
                "value": value,
                "norm": value.lower(),
                "start": match.start(),
                "end": match.end(),
            }
        )
    return tokens


def _merge_token_ranges(ranges: list[tuple[int, int]], max_gap_tokens: int = 1) -> list[tuple[int, int]]:
    if not ranges:
        return []

    merged = [ranges[0]]
    for start, end in ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap_tokens:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _trim_noisy_edges(token_slice: list[dict]) -> list[dict]:
    if not token_slice:
        return token_slice

    def _is_noise(token: dict) -> bool:
        value = token.get("value", "")
        if re.fullmatch(r"\W", value):
            return True
        return len(value) == 1 and not re.search(r"\d", value)

    left = 0
    right = len(token_slice)
    while left < right and _is_noise(token_slice[left]):
        left += 1
    while right > left and _is_noise(token_slice[right - 1]):
        right -= 1
    return token_slice[left:right] if left < right else token_slice


def _build_minimal_changed_spans(text_a: str, text_b: str) -> dict:
    stats = _token_change_stats(text_a, text_b)
    tokens_a = stats["tokens_a"]
    tokens_b = stats["tokens_b"]
    norms_a = stats["norms_a"]
    norms_b = stats["norms_b"]

    matcher = SequenceMatcher(None, norms_a, norms_b)
    ranges_a: list[tuple[int, int]] = []
    ranges_b: list[tuple[int, int]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete") and i1 < i2:
            ranges_a.append((i1, i2))
        if tag in ("replace", "insert") and j1 < j2:
            ranges_b.append((j1, j2))

    ranges_a = _merge_token_ranges(ranges_a, max_gap_tokens=0)
    ranges_b = _merge_token_ranges(ranges_b, max_gap_tokens=0)

    def _to_spans(text: str, tokens: list[dict], ranges: list[tuple[int, int]]) -> list[dict]:
        spans = []
        for start_idx, end_idx in ranges:
            token_slice = _trim_noisy_edges(tokens[start_idx:end_idx])
            if not token_slice:
                continue
            start_char = token_slice[0]["start"]
            end_char = token_slice[-1]["end"]
            changed_text = (text or "")[start_char:end_char].strip()
            if not changed_text:
                continue
            spans.append(
                {
                    "start": start_char,
                    "end": end_char,
                    "text": changed_text,
                    "token_count": len(token_slice),
                }
            )
        return spans

    return {
        "spans_a": _to_spans(text_a, tokens_a, ranges_a),
        "spans_b": _to_spans(text_b, tokens_b, ranges_b),
    }


def generate_word_diff(text_a: str, text_b: str) -> list:
    words_a = [token["value"] for token in _tokenize_with_spans(text_a)]
    words_b = [token["value"] for token in _tokenize_with_spans(text_b)]
    diff = list(difflib.ndiff(words_a, words_b))

    result = []
    for row in diff:
        op = row[0]
        word = row[2:]
        if op == " ":
            result.append({"type": "equal", "value": word})
        elif op == "-":
            result.append({"type": "removed", "value": word})
        elif op == "+":
            result.append({"type": "added", "value": word})
    return result


def _get_all_chunks_by_doc(db: LegalVectorDB, doc_id: str) -> List[dict]:
    storage = db.db._NanoVectorDB__storage
    all_items = storage.get("data", [])
    result = []
    for item in all_items:
        meta = item.get("metadata", {})
        if meta.get("doc_id") != doc_id:
            continue
        chunk = item.get("content", {})
        if isinstance(chunk, dict):
            article = str(chunk.get("article", "")).strip()
            content_text = chunk.get("content", "") or chunk.get("retrieval_text", "")
        else:
            article = ""
            content_text = str(chunk)
        result.append(
            {
                "article": article,
                "clause_index": meta.get("clause_index", 0),
                "content": content_text,
                "metadata": meta,
                "doc_id": meta.get("doc_id"),
                "source_hash": meta.get("source_hash"),
                "source_file": meta.get("source_file"),
                "preview_pdf_path": meta.get("preview_pdf_path") or meta.get("source_file"),
                "page_start": chunk.get("page_start") or meta.get("page_start"),
                "page_end": chunk.get("page_end") or meta.get("page_end"),
                "source_anchors": chunk.get("source_anchors", []),
            }
        )
    return result


def _group_by_article(chunks: List[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for chunk in chunks:
        article = str(chunk.get("article", "")).strip()
        if not article:
            continue
        grouped.setdefault(article, []).append(chunk)
    return grouped


def _merge_chunks_text(chunks: List[dict]) -> str:
    sorted_chunks = sorted(chunks, key=lambda item: item.get("clause_index", 0))
    return "\n".join(chunk.get("content", "").strip() for chunk in sorted_chunks if chunk.get("content", "").strip())


def _merge_chunk_provenance(chunks: List[dict]) -> dict:
    sorted_chunks = sorted(chunks, key=lambda item: item.get("clause_index", 0))
    anchors = []
    pages = []
    preview_pdf_path = None
    source_file = None
    source_hash = None
    doc_id = None

    for chunk in sorted_chunks:
        anchors.extend(chunk.get("source_anchors", []) or [])
        preview_pdf_path = preview_pdf_path or chunk.get("preview_pdf_path") or chunk.get("metadata", {}).get("preview_pdf_path")
        source_file = source_file or chunk.get("source_file") or chunk.get("metadata", {}).get("source_file")
        source_hash = source_hash or chunk.get("source_hash") or chunk.get("metadata", {}).get("source_hash")
        doc_id = doc_id or chunk.get("doc_id") or chunk.get("metadata", {}).get("doc_id")
        if chunk.get("page_start") is not None:
            pages.append(chunk.get("page_start"))
        if chunk.get("page_end") is not None:
            pages.append(chunk.get("page_end"))

    return {
        "doc_id": doc_id,
        "source_hash": source_hash,
        "source_file": source_file,
        "preview_pdf_path": preview_pdf_path or source_file,
        "page_start": min(pages) if pages else None,
        "page_end": max(pages) if pages else None,
        "anchors": anchors,
    }


def _normalize_anchor_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").lower()
    value = re.sub(r"\s+", " ", value).strip()
    return re.sub(r"[^\w\s%./-]", "", value)


def _resolve_citation_anchor(citation: str, provenance: dict) -> Optional[dict]:
    anchors = provenance.get("anchors", []) or []
    if not citation or not anchors:
        return None

    needle = _normalize_anchor_text(citation)
    if not needle:
        return None

    best_match = None
    best_score = -1
    for index, _anchor in enumerate(anchors):
        window = anchors[index:index + 3]
        combined_text = " ".join(item.get("text", "") for item in window)
        haystack = _normalize_anchor_text(combined_text)
        if not haystack:
            continue

        score = 0
        if needle in haystack or haystack in needle:
            score = min(len(needle), len(haystack))
        else:
            score = len(set(needle.split()) & set(haystack.split()))

        if score > best_score:
            best_score = score
            best_match = {
                "doc_id": provenance.get("doc_id"),
                "source_hash": provenance.get("source_hash"),
                "source_file": provenance.get("source_file"),
                "preview_pdf_path": provenance.get("preview_pdf_path"),
                "page_start": window[0].get("page"),
                "page_end": window[-1].get("page"),
                "anchors": window,
            }

    return best_match if best_score > 0 else {
        "doc_id": provenance.get("doc_id"),
        "source_hash": provenance.get("source_hash"),
        "source_file": provenance.get("source_file"),
        "preview_pdf_path": provenance.get("preview_pdf_path"),
        "page_start": provenance.get("page_start"),
        "page_end": provenance.get("page_end"),
        "anchors": anchors[:3],
    }


def _overlap_len(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _merge_bboxes(boxes: list[list[float]]) -> Optional[list[float]]:
    valid = [box for box in boxes if box and len(box) == 4]
    if not valid:
        return None
    return [
        round(min(box[0] for box in valid), 2),
        round(min(box[1] for box in valid), 2),
        round(max(box[2] for box in valid), 2),
        round(max(box[3] for box in valid), 2),
    ]


def _token_set(value: str) -> set[str]:
    return {token for token in re.findall(r"\w+|\d+(?:[.,]\d+)*%?", _normalize_anchor_text(value)) if token}


def _text_match_score(needle: str, haystack: str) -> int:
    if not needle or not haystack:
        return -1
    if needle in haystack:
        return 1000 + len(needle)
    if haystack in needle:
        return 700 + len(haystack)
    overlap = len(_token_set(needle) & _token_set(haystack))
    return overlap * 10


def _build_bbox_from_text_items(anchor: dict, span_text: str) -> Optional[list[float]]:
    text_items = anchor.get("text_items") or []
    if not text_items:
        return None

    needle = _normalize_anchor_text(span_text)
    if not needle:
        return None

    item_payload = []
    for item in text_items:
        norm = _normalize_anchor_text(item.get("text", ""))
        if norm:
            item_payload.append({"norm": norm, "bbox": item.get("bbox")})

    if not item_payload:
        return None

    best_exact_boxes = None
    best_exact_score = None
    best_fuzzy_boxes = None
    best_fuzzy_score = -1
    max_window = min(6, len(item_payload))

    for start_index in range(len(item_payload)):
        for size in range(1, max_window + 1):
            window = item_payload[start_index:start_index + size]
            if not window:
                continue
            combined = " ".join(item["norm"] for item in window).strip()
            if not combined:
                continue

            if needle in combined or combined in needle:
                score = (size, abs(len(combined) - len(needle)))
                if best_exact_score is None or score < best_exact_score:
                    best_exact_score = score
                    best_exact_boxes = [item["bbox"] for item in window if item.get("bbox")]
                continue

            overlap = len(set(needle.split()) & set(combined.split()))
            if overlap > best_fuzzy_score:
                best_fuzzy_score = overlap
                best_fuzzy_boxes = [item["bbox"] for item in window if item.get("bbox")]

    if best_exact_boxes:
        return _merge_bboxes(best_exact_boxes or [])
    if best_fuzzy_score > 0:
        return _merge_bboxes(best_fuzzy_boxes or [])
    return None


def _resolve_window_entries(window: list[dict], span_text: str) -> list[dict]:
    resolved = []
    needle = _normalize_anchor_text(span_text)
    needle_tokens = _token_set(span_text)

    for anchor in window:
        anchor_text = anchor.get("text", "")
        anchor_norm = _normalize_anchor_text(anchor_text)
        if not anchor_norm:
            continue

        bbox = _build_bbox_from_text_items(anchor, span_text)
        if not bbox:
            overlap = len(needle_tokens & _token_set(anchor_text))
            if overlap <= 0 and needle not in anchor_norm and anchor_norm not in needle:
                continue
            bbox = anchor.get("bbox")

        if not bbox:
            continue

        resolved.append(
            {
                "text": span_text,
                "page": anchor.get("page"),
                "block_index": anchor.get("block_index"),
                "line_index": anchor.get("line_index"),
                "bbox": bbox,
            }
        )
    return resolved


def _cleanup_resolved_entries(entries: list[dict]) -> list[dict]:
    if not entries:
        return []

    entries = sorted(
        entries,
        key=lambda item: (
            item.get("page") or 0,
            (item.get("bbox") or [0, 0, 0, 0])[1],
            (item.get("bbox") or [0, 0, 0, 0])[0],
        ),
    )

    cleaned = []
    for entry in entries:
        bbox = entry.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        if not cleaned:
            cleaned.append(entry)
            continue

        prev = cleaned[-1]
        prev_bbox = prev.get("bbox")
        same_page = prev.get("page") == entry.get("page")
        if same_page and prev_bbox:
            vertical_gap = abs(bbox[1] - prev_bbox[1])
            horizontal_gap = max(0, bbox[0] - prev_bbox[2])
            same_line = vertical_gap <= 2
            close_inline = horizontal_gap <= 4
            overlap_inline = bbox[0] <= prev_bbox[2] + 1
            if same_line and (close_inline or overlap_inline):
                prev["bbox"] = _merge_bboxes([prev_bbox, bbox]) or prev_bbox
                continue

        cleaned.append(entry)

    return cleaned


def _resolve_changed_spans_to_anchors(spans: list[dict], provenance: dict) -> list[dict]:
    anchors = provenance.get("anchors", []) or []
    if not spans or not anchors:
        return []

    resolved = []
    for span in spans:
        span_text = span.get("text", "")
        if not span_text:
            continue

        exact_entries = []
        for anchor in anchors:
            exact_entries.extend(_resolve_window_entries([anchor], span_text))
        exact_entries = _cleanup_resolved_entries(exact_entries)
        if exact_entries:
            for entry in exact_entries:
                resolved.append(
                    {
                        **entry,
                        "start": span.get("start", 0),
                        "end": span.get("end", 0),
                        "token_count": span.get("token_count", 0),
                    }
                )
            continue

        best_window = []
        best_score = -1
        for start_index in range(len(anchors)):
            for size in range(1, min(4, len(anchors) - start_index) + 1):
                window = anchors[start_index:start_index + size]
                combined = " ".join(anchor.get("text", "") for anchor in window)
                score = _text_match_score(_normalize_anchor_text(span_text), _normalize_anchor_text(combined))
                if score > best_score:
                    best_score = score
                    best_window = window

        if not best_window:
            continue

        for entry in _cleanup_resolved_entries(_resolve_window_entries(best_window, span_text)):
            resolved.append(
                {
                    **entry,
                    "start": span.get("start", 0),
                    "end": span.get("end", 0),
                    "token_count": span.get("token_count", 0),
                }
            )

    return resolved


def _build_clause_highlights(provenance: dict, strike: bool = False) -> list[dict]:
    anchors = provenance.get("anchors", []) or []
    highlighted = []
    seen = set()
    for anchor in anchors:
        page = anchor.get("page")
        bbox = anchor.get("bbox")
        if page is None or not bbox or len(bbox) != 4:
            continue
        key = (int(page), round(float(bbox[0]), 2), round(float(bbox[1]), 2), round(float(bbox[2]), 2), round(float(bbox[3]), 2))
        if key in seen:
            continue
        seen.add(key)
        highlighted.append(
            {
                "page": int(page),
                "bbox": [round(float(bbox[0]), 2), round(float(bbox[1]), 2), round(float(bbox[2]), 2), round(float(bbox[3]), 2)],
                "block_index": anchor.get("block_index"),
                "line_index": anchor.get("line_index"),
                "kind": "block",
                "strike": bool(strike),
            }
        )
    highlighted.sort(key=lambda item: (item.get("page", 0), item.get("block_index") or 0, item.get("line_index") or 0))
    return highlighted


def _apply_inline_highlight_style(highlights: list[dict], strike: bool = False) -> list[dict]:
    return [{**item, "kind": "inline", "strike": bool(strike)} for item in highlights]


def _should_use_block_highlight_for_modified(is_numeric: bool, token_stats: dict, highlight_payload: dict) -> bool:
    if is_numeric:
        return False
    total_tokens = max(1, len(token_stats.get("tokens_a", [])), len(token_stats.get("tokens_b", [])))
    changed_tokens = max(token_stats.get("changed_a", 0), token_stats.get("changed_b", 0))
    coverage = changed_tokens / total_tokens
    spans_count = max(
        len(highlight_payload.get("changed_spans_a", [])),
        len(highlight_payload.get("changed_spans_b", [])),
    )
    return coverage >= 0.36 or spans_count >= 5


def _is_minor_wording_change(text_a: str, text_b: str) -> bool:
    if not text_a or not text_b:
        return False
    if _extract_numeric_signals(text_a) != _extract_numeric_signals(text_b):
        return False

    stats = _token_change_stats(text_a, text_b)
    if not stats["tokens_a"] or not stats["tokens_b"]:
        return False

    changed_tokens = max(stats["changed_a"], stats["changed_b"])
    if stats["ratio"] >= 0.96 and changed_tokens <= 6:
        return True
    if stats["ratio"] >= 0.92 and changed_tokens <= 3:
        return True
    return False


def _build_highlight_payload(text_a: str, text_b: str, provenance_a: dict, provenance_b: dict) -> dict:
    changed = _build_minimal_changed_spans(text_a, text_b)
    spans_a = changed.get("spans_a", [])
    spans_b = changed.get("spans_b", [])
    highlights_a = _resolve_changed_spans_to_anchors(spans_a, provenance_a)
    highlights_b = _resolve_changed_spans_to_anchors(spans_b, provenance_b)

    citation_a = spans_a[0]["text"] if spans_a else ""
    citation_b = spans_b[0]["text"] if spans_b else ""
    return {
        "word_diff": generate_word_diff(text_a, text_b),
        "changed_spans_a": spans_a,
        "changed_spans_b": spans_b,
        "highlight_anchors_a": highlights_a,
        "highlight_anchors_b": highlights_b,
        "citation_a": citation_a,
        "citation_b": citation_b,
        "citation_anchor_a": _resolve_citation_anchor(citation_a, provenance_a) if citation_a else None,
        "citation_anchor_b": _resolve_citation_anchor(citation_b, provenance_b) if citation_b else None,
    }


def extract_numbers(text: str) -> list[str]:
    return sorted(_NUMERIC_REGEX.findall(text or ""))


def has_numeric_diff(text_a: str, text_b: str) -> bool:
    return _extract_numeric_signals(text_a) != _extract_numeric_signals(text_b)


def normalize_for_compare(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    return re.sub(r"\s+", " ", text).strip().lower()


def run_comparison(
    doc_id_a: str,
    doc_id_b: str,
    db_path: str = "legal_data.json",
    model: str = DEFAULT_MODEL,
    verbose: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> List[dict]:
    if progress_callback:
        progress_callback(5, "Đang khởi tạo so sánh")

    db = LegalVectorDB(db_name=db_path)
    chunks_a = _get_all_chunks_by_doc(db, doc_id_a)
    chunks_b = _get_all_chunks_by_doc(db, doc_id_b)

    if not chunks_a:
        raise ValueError(f"Không tìm thấy dữ liệu cho doc_id='{doc_id_a}'. Hãy ingest file trước.")
    if not chunks_b:
        raise ValueError(f"Không tìm thấy dữ liệu cho doc_id='{doc_id_b}'. Hãy ingest file trước.")

    grouped_a = _group_by_article(chunks_a)
    grouped_b = _group_by_article(chunks_b)

    def article_sort_key(label: str):
        nums = [float(n) for n in re.findall(r"\d+", label)]
        return tuple(nums) if nums else (999,)

    all_articles = sorted(set(grouped_a.keys()) | set(grouped_b.keys()), key=article_sort_key)

    if verbose:
        print(f"[Comparator] So sánh {len(all_articles)} điều/khoản")

    text_a_map = {article: _merge_chunks_text(chunks) for article, chunks in grouped_a.items()}
    text_b_map = {article: _merge_chunks_text(chunks) for article, chunks in grouped_b.items()}
    prov_a_map = {article: _merge_chunk_provenance(chunks) for article, chunks in grouped_a.items()}
    prov_b_map = {article: _merge_chunk_provenance(chunks) for article, chunks in grouped_b.items()}

    vec_a_map: Dict[str, np.ndarray] = {}
    vec_b_map: Dict[str, np.ndarray] = {}
    norm_a_map: Dict[str, float] = {}
    norm_b_map: Dict[str, float] = {}

    def get_vec_a(article: str):
        if article not in vec_a_map:
            vec = db.embedder.get_embeddings([text_a_map[article]])[0]
            vec_a_map[article] = vec
            norm_a_map[article] = float(np.linalg.norm(vec))
        return vec_a_map[article], norm_a_map[article]

    def get_vec_b(article: str):
        if article not in vec_b_map:
            vec = db.embedder.get_embeddings([text_b_map[article]])[0]
            vec_b_map[article] = vec
            norm_b_map[article] = float(np.linalg.norm(vec))
        return vec_b_map[article], norm_b_map[article]

    results: List[Optional[dict]] = []
    pending_llm_tasks = []
    total_articles = max(1, len(all_articles))

    for article_index, article in enumerate(all_articles, start=1):
        if progress_callback:
            pct = 10 + int((article_index - 1) / total_articles * 75)
            progress_callback(pct, f"Đang so sánh {article_index}/{total_articles} điều khoản")

        in_a = article in grouped_a
        in_b = article in grouped_b

        text_a = text_a_map.get(article, "") if in_a else ""
        text_b = text_b_map.get(article, "") if in_b else ""
        provenance_a = prov_a_map.get(article) if in_a else {"page_start": None, "page_end": None, "anchors": []}
        provenance_b = prov_b_map.get(article) if in_b else {"page_start": None, "page_end": None, "anchors": []}
        article_label = article if "Dieu" in article or "Điều" in article else f"Dieu {article}"

        if not in_a and in_b:
            block_highlights_b = _build_clause_highlights(provenance_b, strike=False)
            results.append(
                {
                    "article": article,
                    "status": "ADDED",
                    "summary": f"{article_label} chỉ xuất hiện trong bản B. Đây là điều khoản được thêm mới.",
                    "exact_difference": "Toàn bộ điều khoản",
                    "change_type": "Thêm mới",
                    "citation_a": "",
                    "citation_b": text_b[:300] + ("..." if len(text_b) > 300 else ""),
                    "text_a": "",
                    "text_b": text_b,
                    "pdf_anchor_a": None,
                    "pdf_anchor_b": provenance_b,
                    "citation_anchor_a": None,
                    "citation_anchor_b": _resolve_citation_anchor(text_b[:300], provenance_b),
                    "highlight_anchors_a": [],
                    "highlight_anchors_b": block_highlights_b,
                    "changed_spans_a": [],
                    "changed_spans_b": [],
                    "highlight_mode": "block",
                }
            )
            continue

        if in_a and not in_b:
            vec_a, norm_a = get_vec_a(article)
            best_match_article = None
            best_sim = 0.0
            best_text_b = ""
            best_provenance_b = None

            for article_b, candidate_text_b in text_b_map.items():
                if not candidate_text_b.strip():
                    continue
                vec_b, norm_b = get_vec_b(article_b)
                sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b)) if (norm_a > 0 and norm_b > 0) else 0.0
                if sim > best_sim:
                    best_sim = sim
                    best_match_article = article_b if "Dieu" in article_b or "Điều" in article_b else f"Dieu {article_b}"
                    best_text_b = candidate_text_b
                    best_provenance_b = prov_b_map[article_b]

            if best_sim >= 0.85 and best_provenance_b:
                highlight_payload = _build_highlight_payload(text_a, best_text_b, provenance_a, best_provenance_b)
                moved_highlights_a = _build_clause_highlights(provenance_a, strike=True)
                moved_highlights_b = _build_clause_highlights(best_provenance_b, strike=False)
                results.append(
                    {
                        "article": article,
                        "status": "MOVED",
                        "summary": f"{article_label} đã được chuyển sang {best_match_article} ở bản mới (độ tương đồng: {best_sim:.2f}).",
                        "exact_difference": "Chuyển vị trí điều khoản",
                        "change_type": "Thay đổi vị trí",
                        "citation_a": highlight_payload["citation_a"],
                        "citation_b": highlight_payload["citation_b"],
                        "text_a": text_a,
                        "text_b": best_text_b,
                        "word_diff": highlight_payload["word_diff"],
                        "changed_spans_a": highlight_payload["changed_spans_a"],
                        "changed_spans_b": highlight_payload["changed_spans_b"],
                        "highlight_anchors_a": moved_highlights_a,
                        "highlight_anchors_b": moved_highlights_b,
                        "pdf_anchor_a": provenance_a,
                        "pdf_anchor_b": best_provenance_b,
                        "citation_anchor_a": highlight_payload["citation_anchor_a"],
                        "citation_anchor_b": highlight_payload["citation_anchor_b"],
                        "highlight_mode": "block",
                    }
                )
            else:
                removed_highlights_a = _build_clause_highlights(provenance_a, strike=True)
                results.append(
                    {
                        "article": article,
                        "status": "REMOVED",
                        "summary": f"{article_label} chỉ xuất hiện trong bản A. Điều khoản này đã bị xóa bỏ.",
                        "exact_difference": "Toàn bộ điều khoản",
                        "change_type": "Xóa bỏ",
                        "citation_a": text_a[:300] + ("..." if len(text_a) > 300 else ""),
                        "citation_b": "",
                        "text_a": text_a,
                        "text_b": "",
                        "pdf_anchor_a": provenance_a,
                        "pdf_anchor_b": None,
                        "citation_anchor_a": _resolve_citation_anchor(text_a[:300], provenance_a),
                        "citation_anchor_b": None,
                        "highlight_anchors_a": removed_highlights_a,
                        "highlight_anchors_b": [],
                        "changed_spans_a": [],
                        "changed_spans_b": [],
                        "highlight_mode": "block",
                    }
                )
            continue

        if normalize_for_compare(text_a) == normalize_for_compare(text_b):
            results.append(
                {
                    "article": article,
                    "status": "IDENTICAL",
                    "summary": "Nội dung hai bản giống nhau hoàn toàn.",
                    "exact_difference": "",
                    "change_type": "",
                    "citation_a": "",
                    "citation_b": "",
                    "text_a": text_a,
                    "text_b": text_b,
                    "pdf_anchor_a": provenance_a,
                    "pdf_anchor_b": provenance_b,
                    "citation_anchor_a": None,
                    "citation_anchor_b": None,
                    "highlight_anchors_a": [],
                    "highlight_anchors_b": [],
                    "changed_spans_a": [],
                    "changed_spans_b": [],
                }
            )
            continue

        token_stats = _token_change_stats(text_a, text_b)
        is_numeric = has_numeric_diff(text_a, text_b)
        is_minor = _is_minor_wording_change(text_a, text_b)
        lexical_sim = token_stats["ratio"]
        semantic_sim = 0.0

        if not is_numeric and not is_minor and lexical_sim >= 0.93:
            semantic_sim = lexical_sim
        elif not is_numeric and not is_minor:
            vec_a, norm_a = get_vec_a(article)
            vec_b, norm_b = get_vec_b(article)
            semantic_sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b)) if (norm_a > 0 and norm_b > 0) else 0.0

        if is_minor:
            results.append(
                {
                    "article": article,
                    "status": "MINOR-MODIFIED",
                    "summary": "Thay đổi chỉ là diễn đạt lại câu chữ nên được bỏ qua trong báo cáo.",
                    "exact_difference": "Thay đổi nhỏ về câu chữ, không làm đổi ý chính.",
                    "change_type": "Thay đổi nhỏ về câu chữ",
                    "impact_level": "Nhỏ",
                    "citation_a": "",
                    "citation_b": "",
                    "text_a": text_a,
                    "text_b": text_b,
                    "word_diff": [],
                    "changed_spans_a": [],
                    "changed_spans_b": [],
                    "highlight_anchors_a": [],
                    "highlight_anchors_b": [],
                    "pdf_anchor_a": provenance_a,
                    "pdf_anchor_b": provenance_b,
                    "citation_anchor_a": None,
                    "citation_anchor_b": None,
                    "highlight_mode": "none",
                }
            )
            continue

        if not is_numeric and semantic_sim >= 0.93:
            highlight_payload = _build_highlight_payload(text_a, text_b, provenance_a, provenance_b)
            use_block = _should_use_block_highlight_for_modified(is_numeric, token_stats, highlight_payload)
            if use_block:
                highlights_a = _build_clause_highlights(provenance_a, strike=True)
                highlights_b = _build_clause_highlights(provenance_b, strike=False)
                highlight_mode = "block"
            else:
                highlights_a = _apply_inline_highlight_style(highlight_payload["highlight_anchors_a"], strike=True)
                highlights_b = _apply_inline_highlight_style(highlight_payload["highlight_anchors_b"], strike=False)
                highlight_mode = "inline"
            results.append(
                {
                    "article": article,
                    "status": "MODIFIED",
                    "summary": "Thay đổi đáng chú ý ở cách diễn đạt hoặc nội dung.",
                    "exact_difference": "Nội dung đã được chỉnh sửa ở mức có ý nghĩa.",
                    "change_type": "Sửa nội dung",
                    "impact_level": "Trung bình",
                    "citation_a": highlight_payload["citation_a"],
                    "citation_b": highlight_payload["citation_b"],
                    "text_a": text_a,
                    "text_b": text_b,
                    "word_diff": highlight_payload["word_diff"],
                    "changed_spans_a": highlight_payload["changed_spans_a"],
                    "changed_spans_b": highlight_payload["changed_spans_b"],
                    "highlight_anchors_a": highlights_a,
                    "highlight_anchors_b": highlights_b,
                    "pdf_anchor_a": provenance_a,
                    "pdf_anchor_b": provenance_b,
                    "citation_anchor_a": highlight_payload["citation_anchor_a"],
                    "citation_anchor_b": highlight_payload["citation_anchor_b"],
                    "highlight_mode": highlight_mode,
                }
            )
            continue

        pending_llm_tasks.append(
            {
                "article": article,
                "article_label": article_label,
                "text_a": text_a,
                "text_b": text_b,
                "is_numeric": is_numeric,
                "token_stats": token_stats,
                "result_index": len(results),
                "provenance_a": provenance_a,
                "provenance_b": provenance_b,
            }
        )
        results.append(None)

    if pending_llm_tasks:
        if progress_callback:
            progress_callback(88, f"Đang phân tích {len(pending_llm_tasks)} điều khoản bằng LLM")

        def _run_llm(task):
            return task, compare_clauses(
                task["text_a"],
                task["text_b"],
                task["article_label"],
                model=model,
                has_numeric_change=task["is_numeric"],
            )

        max_workers = min(2, len(pending_llm_tasks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_llm, task): task for task in pending_llm_tasks}
            completed = 0
            total_pending = max(1, len(pending_llm_tasks))

            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                try:
                    _, llm_result = future.result()
                except Exception as exc:
                    llm_result = {"status": "ERROR", "summary": f"Lỗi gọi LLM đâ luồng: {exc}"}

                completed += 1
                if progress_callback:
                    llm_progress = 88 + int(completed / total_pending * 10)
                    progress_callback(min(98, llm_progress), f"LLM {completed}/{total_pending} điều khoản")

                highlight_payload = _build_highlight_payload(
                    task["text_a"],
                    task["text_b"],
                    task["provenance_a"],
                    task["provenance_b"],
                )
                status = llm_result.get("status", "ERROR")
                change_type = llm_result.get("change_type", "")
                if task["is_numeric"] and status not in {"ADDED", "REMOVED", "MOVED"}:
                    status = "MODIFIED"
                    if not change_type or "nhỏ" in change_type.lower() or "câu chữ" in change_type.lower():
                        change_type = "Thay đổi số liệu"
                if status == "MODIFIED":
                    use_block = _should_use_block_highlight_for_modified(
                        task["is_numeric"],
                        task.get("token_stats") or _token_change_stats(task["text_a"], task["text_b"]),
                        highlight_payload,
                    )
                    if use_block:
                        highlights_a = _build_clause_highlights(task["provenance_a"], strike=True)
                        highlights_b = _build_clause_highlights(task["provenance_b"], strike=False)
                        highlight_mode = "block"
                    else:
                        highlights_a = _apply_inline_highlight_style(highlight_payload["highlight_anchors_a"], strike=True)
                        highlights_b = _apply_inline_highlight_style(highlight_payload["highlight_anchors_b"], strike=False)
                        highlight_mode = "inline"
                elif status == "REMOVED":
                    highlights_a = _build_clause_highlights(task["provenance_a"], strike=True)
                    highlights_b = []
                    highlight_mode = "block"
                elif status == "ADDED":
                    highlights_a = []
                    highlights_b = _build_clause_highlights(task["provenance_b"], strike=False)
                    highlight_mode = "block"
                elif status == "MOVED":
                    highlights_a = _build_clause_highlights(task["provenance_a"], strike=True)
                    highlights_b = _build_clause_highlights(task["provenance_b"], strike=False)
                    highlight_mode = "block"
                else:
                    highlights_a = _apply_inline_highlight_style(highlight_payload["highlight_anchors_a"], strike=False)
                    highlights_b = _apply_inline_highlight_style(highlight_payload["highlight_anchors_b"], strike=False)
                    highlight_mode = "inline"
                results[task["result_index"]] = {
                    "article": task["article"],
                    "status": status,
                    "summary": llm_result.get("summary", ""),
                    "exact_difference": llm_result.get("exact_difference", ""),
                    "change_type": change_type,
                    "citation_a": highlight_payload["citation_a"] or llm_result.get("citation_a", ""),
                    "citation_b": highlight_payload["citation_b"] or llm_result.get("citation_b", ""),
                    "text_a": task["text_a"],
                    "text_b": task["text_b"],
                    "word_diff": highlight_payload["word_diff"],
                    "changed_spans_a": highlight_payload["changed_spans_a"],
                    "changed_spans_b": highlight_payload["changed_spans_b"],
                    "highlight_anchors_a": highlights_a,
                    "highlight_anchors_b": highlights_b,
                    "pdf_anchor_a": task["provenance_a"],
                    "pdf_anchor_b": task["provenance_b"],
                    "citation_anchor_a": highlight_payload["citation_anchor_a"]
                    or _resolve_citation_anchor(llm_result.get("citation_a", ""), task["provenance_a"]),
                    "citation_anchor_b": highlight_payload["citation_anchor_b"]
                    or _resolve_citation_anchor(llm_result.get("citation_b", ""), task["provenance_b"]),
                    "highlight_mode": highlight_mode,
                }

    for entry in results:
        if not entry:
            continue
        pdf_anchor_a = entry.get("pdf_anchor_a") or {}
        pdf_anchor_b = entry.get("pdf_anchor_b") or {}
        entry.setdefault("doc_id_a", pdf_anchor_a.get("doc_id") or doc_id_a)
        entry.setdefault("doc_id_b", pdf_anchor_b.get("doc_id") or doc_id_b)
        entry.setdefault("preview_pdf_path_a", pdf_anchor_a.get("preview_pdf_path"))
        entry.setdefault("preview_pdf_path_b", pdf_anchor_b.get("preview_pdf_path"))
        entry.setdefault("source_hash_a", pdf_anchor_a.get("source_hash"))
        entry.setdefault("source_hash_b", pdf_anchor_b.get("source_hash"))

    if progress_callback:
        progress_callback(100, "So sanh hoan tat")

    return [entry for entry in results if entry]


def save_report(results: List[dict], output_path: str = "comparison_report.json"):
    skip_statuses = {"IDENTICAL", "MINOR-MODIFIED"}
    skip_fields = {"word_diff"}
    filtered = [
        {key: value for key, value in result.items() if key not in skip_fields}
        for result in results
        if result.get("status") not in skip_statuses
    ]

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(filtered, handle, ensure_ascii=False, indent=2)


def print_report(results: List[dict]):
    added = [item for item in results if item["status"] == "ADDED"]
    removed = [item for item in results if item["status"] == "REMOVED"]
    moved = [item for item in results if item["status"] == "MOVED"]
    modified = [item for item in results if item["status"] == "MODIFIED"]
    identical = [item for item in results if item["status"] == "IDENTICAL"]
    errors = [item for item in results if item["status"] == "ERROR"]

    print("=" * 70)
    print("KẾT QUẢ SO SÁNH")
    print("=" * 70)
    print(f"Thêm mới : {len(added)}")
    print(f"Xóa bỏ   : {len(removed)}")
    print(f"Chuyển vị trí: {len(moved)}")
    print(f"Sửa đổi  : {len(modified)}")
    print(f"Giống nhau: {len(identical)}")
    if errors:
        print(f"Lỗi      : {len(errors)}")

    for item in added + removed + moved + modified:
        print("-" * 70)
        print(f"{item.get('article')} [{item.get('status')}]")
        print(item.get("summary", ""))
