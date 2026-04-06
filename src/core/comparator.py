"""
comparator.py
Pipeline so sánh 2 văn bản hợp đồng theo từng Điều khoản.

Logic chính:
1. Lấy toàn bộ chunks của doc_a và doc_b từ VectorDB
2. Ghép cặp các Điều có cùng số thứ tự
3. Điều chỉ có ở 1 bản → ADDED / REMOVED
4. Điều có ở cả 2 bản → đưa qua LLM phân tích → IDENTICAL / MODIFIED
"""

import json
import re
import difflib
import numpy as np
from typing import Dict, List, Optional
from src.database.vector_store import LegalVectorDB
from src.core.llm import compare_clauses, DEFAULT_MODEL

def generate_word_diff(text_a: str, text_b: str) -> list:
    """Tạo mảng diff ở cấp độ từ (word-level)."""
    words_a = text_a.split()
    words_b = text_b.split()
    diff = list(difflib.ndiff(words_a, words_b))
    
    result = []
    for d in diff:
        op = d[0]
        word = d[2:]
        if op == " ":
            result.append({"type": "equal", "value": word})
        elif op == "-":
            result.append({"type": "removed", "value": word})
        elif op == "+":
            result.append({"type": "added", "value": word})
    return result


# ─────────────────────────────────────────────
# Lấy dữ liệu từ VectorDB
# ─────────────────────────────────────────────

def _get_all_chunks_by_doc(db: LegalVectorDB, doc_id: str) -> List[dict]:
    """
    Lay toan bo chunks thuoc ve doc_id tu NanoVectorDB.
    NanoVectorDB luu du lieu trong __storage['data'] (list of dicts).
    Moi item co dang: {'__id__': ..., 'content': <chunk_dict>, 'metadata': {'doc_id': ...}}
    """
    storage = db.db._NanoVectorDB__storage  # name-mangled private attr
    all_items = storage.get("data", [])
    result = []
    for item in all_items:
        meta = item.get("metadata", {})
        # Filter by metadata.doc_id (correct), NOT content.doc_id (may be hardcoded)
        if meta.get("doc_id") != doc_id:
            continue
        chunk = item.get("content", {})
        # chunk is a dict with 'article', 'content' (text), etc.
        if isinstance(chunk, dict):
            article = str(chunk.get("article", "")).strip()
            content_text = chunk.get("content", "") or chunk.get("retrieval_text", "")
        else:
            article = ""
            content_text = str(chunk)
        result.append({
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
        })
    return result


def _group_by_article(chunks: List[dict]) -> Dict[str, List[dict]]:
    """
    Nhóm chunks theo số Điều (article).
    Ví dụ: { "2": [chunk1, chunk2], "5": [chunk3] }
    """
    grouped: Dict[str, List[dict]] = {}
    for chunk in chunks:
        article = str(chunk.get("article", "")).strip()
        if not article:
            continue
        if article not in grouped:
            grouped[article] = []
        grouped[article].append(chunk)
    return grouped


def _merge_chunks_text(chunks: List[dict]) -> str:
    """Ghép toàn bộ khoản của một Điều thành 1 đoạn text."""
    # Sắp xếp theo clause_index để đúng thứ tự
    sorted_chunks = sorted(chunks, key=lambda c: c.get("clause_index", 0))
    parts = []
    for c in sorted_chunks:
        content = c.get("content", "").strip()
        if content:
            parts.append(content)
    return "\n".join(parts)


def _merge_chunk_provenance(chunks: List[dict]) -> dict:
    sorted_chunks = sorted(chunks, key=lambda c: c.get("clause_index", 0))
    anchors = []
    pages = []
    preview_pdf_path = None
    source_file = None
    source_hash = None
    doc_id = None

    for chunk in sorted_chunks:
        chunk_anchors = chunk.get("source_anchors", []) or []
        anchors.extend(chunk_anchors)
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
    value = value or ""
    value = re.sub(r"\s+", " ", value.lower()).strip()
    value = re.sub(r"[^\w\s%./-]", "", value)
    return value


def _resolve_citation_anchor(citation: str, provenance: dict) -> Optional[dict]:
    anchors = provenance.get("anchors", []) or []
    if not citation or not anchors:
        return None

    needle = _normalize_anchor_text(citation)
    if not needle:
        return None

    best_match = None
    best_score = -1

    for index, anchor in enumerate(anchors):
        window = anchors[index:index + 3]
        combined_text = " ".join(item.get("text", "") for item in window)
        haystack = _normalize_anchor_text(combined_text)
        if not haystack:
            continue

        score = 0
        if needle in haystack or haystack in needle:
            score = min(len(needle), len(haystack))
        else:
            common = set(needle.split()) & set(haystack.split())
            score = len(common)

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


# ─────────────────────────────────────────────
# Utils So Sánh Nâng Cao
# ─────────────────────────────────────────────
import unicodedata
from difflib import SequenceMatcher

_NUMERIC_REGEX = re.compile(r'\b\d+(?:[.,]\d+)*\b')

def extract_numbers(text: str) -> list[str]:
    """Tìm tất cả số nguyên và số thập phân (có dấu phân cách , hoặc .)"""
    return sorted(_NUMERIC_REGEX.findall(text))

def has_numeric_diff(text_a: str, text_b: str) -> bool:
    """So sánh tập hợp các số giữa 2 đoạn văn để phát hiện thay đổi định lượng"""
    return extract_numbers(text_a) != extract_numbers(text_b)

def normalize_for_compare(t: str) -> str:
    """Loại bỏ ký tự thừa, normalize chuẩn Unicode"""
    t = unicodedata.normalize('NFC', t)
    return re.sub(r'\s+', ' ', t).strip().lower()

# ─────────────────────────────────────────────
# Pipeline chính
# ─────────────────────────────────────────────

def run_comparison(
    doc_id_a: str,
    doc_id_b: str,
    db_path: str = "legal_data.json",
    model: str = DEFAULT_MODEL,
    verbose: bool = True,
) -> List[dict]:
    """
    So sánh 2 văn bản hợp đồng và trả về danh sách kết quả.

    Returns:
    [
        {
            "article": "5",
            "status": "MODIFIED",            # ADDED | REMOVED | MODIFIED | IDENTICAL
            "summary": "...",
            "citation_a": "...",
            "citation_b": "...",
            "text_a": "...",                  # Toàn bộ text Điều trong bản A
            "text_b": "...",                  # Toàn bộ text Điều trong bản B
        },
        ...
    ]
    """
    if verbose:
        print(f"[Comparator] Đang load VectorDB: {db_path}")

    db = LegalVectorDB(db_name=db_path)

    if verbose:
        print(f"[Comparator] Lấy chunks của '{doc_id_a}' và '{doc_id_b}'")

    chunks_a = _get_all_chunks_by_doc(db, doc_id_a)
    chunks_b = _get_all_chunks_by_doc(db, doc_id_b)

    if not chunks_a:
        raise ValueError(f"Không tìm thấy dữ liệu cho doc_id='{doc_id_a}'. Hãy ingest file trước!")
    if not chunks_b:
        raise ValueError(f"Không tìm thấy dữ liệu cho doc_id='{doc_id_b}'. Hãy ingest file trước!")

    grouped_a = _group_by_article(chunks_a)
    grouped_b = _group_by_article(chunks_b)

    def article_sort_key(label: str):
        # Extract all numbers from string to sort correctly (e.g., "Chương 2 > Điều 1" -> (2, 1))
        nums = [float(n) for n in re.findall(r'\d+', label)]
        return tuple(nums) if nums else (999,)

    all_articles = sorted(
        set(grouped_a.keys()) | set(grouped_b.keys()),
        key=article_sort_key
    )

    if verbose:
        total = len(all_articles)
        only_a = len(set(grouped_a) - set(grouped_b))
        only_b = len(set(grouped_b) - set(grouped_a))
        shared = len(set(grouped_a) & set(grouped_b))
        print(f"\n{'─'*50}")
        print(f"Tổng số Điều cần xét  : {total}")
        print(f"  Chỉ có trong bản A  : {only_a}")
        print(f"  Chỉ có trong bản B  : {only_b}")
        print(f"  Có ở cả 2 bản       : {shared}")
        print(f"{'─'*50}\n")

    import concurrent.futures

    results = []
    pending_llm_tasks = []

    if verbose:
        print("[Comparator] Sử dụng Lazy Embedding Caching để tăng tốc độ so sánh...")
    
    text_a_map = {art: _merge_chunks_text(chunks) for art, chunks in grouped_a.items()}
    text_b_map = {art: _merge_chunks_text(chunks) for art, chunks in grouped_b.items()}
    
    prov_a_map = {art: _merge_chunk_provenance(chunks) for art, chunks in grouped_a.items()}
    prov_b_map = {art: _merge_chunk_provenance(chunks) for art, chunks in grouped_b.items()}
    
    vec_a_map, norm_a_map = {}, {}
    vec_b_map, norm_b_map = {}, {}

    def get_vec_a(art):
        if art not in vec_a_map:
            vec = db.embedder.get_embeddings([text_a_map[art]])[0]
            vec_a_map[art] = vec
            norm_a_map[art] = np.linalg.norm(vec)
        return vec_a_map[art], norm_a_map[art]

    def get_vec_b(art):
        if art not in vec_b_map:
            vec = db.embedder.get_embeddings([text_b_map[art]])[0]
            vec_b_map[art] = vec
            norm_b_map[art] = np.linalg.norm(vec)
        return vec_b_map[art], norm_b_map[art]

    for article in all_articles:
        in_a = article in grouped_a
        in_b = article in grouped_b

        text_a = text_a_map.get(article, "") if in_a else ""
        text_b = text_b_map.get(article, "") if in_b else ""
        provenance_a = prov_a_map.get(article) if in_a else {"page_start": None, "page_end": None, "anchors": []}
        provenance_b = prov_b_map.get(article) if in_b else {"page_start": None, "page_end": None, "anchors": []}
        article_label = article if "Điều" in article else f"Điều {article}"

        # ── Case 1: Chỉ có ở bản B → BẢN MỚI THÊM VÀO
        if not in_a and in_b:
            entry = {
                "article": article,
                "status": "ADDED",
                "summary": f"{article_label} chỉ xuất hiện trong bản B (bản mới). Đây là điều khoản được thêm mới.",
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
            }
            results.append(entry)
            if verbose:
                print(f"  [ADDED   ] {article_label}")
            continue

        # ── Case 2: Chỉ có ở bản A → ĐÃ BỊ XÓA HOẶC DI CHUYỂN
        if in_a and not in_b:
            vec_a, norm_a = get_vec_a(article)
            
            best_match_article = None
            best_sim = 0.0
            best_text_b = ""
            best_provenance_b = None
            
            for art_b, txt_b in text_b_map.items():
                if not txt_b.strip(): continue
                vec_b, norm_b = get_vec_b(art_b)
                sim = np.dot(vec_a, vec_b) / (norm_a * norm_b) if (norm_a > 0 and norm_b > 0) else 0.0
                
                if sim > best_sim:
                    best_sim = sim
                    best_match_article = art_b if "Điều" in art_b else f"Điều {art_b}"
                    best_text_b = txt_b
                    best_provenance_b = prov_b_map[art_b]
            
            if best_sim >= 0.85:
                entry = {
                    "article": article,
                    "status": "MOVED",
                    "summary": f"{article_label} đã được chuyển sang {best_match_article} ở bản mới (Độ tương đồng: {best_sim:.2f}).",
                    "exact_difference": "Chuyển vị trí điều khoản",
                    "change_type": "Thay đổi vị trí",
                    "citation_a": text_a[:200] + "...",
                    "citation_b": best_text_b[:200] + "...",
                    "text_a": text_a,
                    "text_b": best_text_b,
                    "word_diff": generate_word_diff(text_a, best_text_b),
                    "pdf_anchor_a": provenance_a,
                    "pdf_anchor_b": best_provenance_b,
                    "citation_anchor_a": _resolve_citation_anchor(text_a[:200], provenance_a),
                    "citation_anchor_b": _resolve_citation_anchor(best_text_b[:200], best_provenance_b or {"anchors": []}),
                }
                results.append(entry)
                if verbose:
                    print(f"  [MOVED   ] {article_label} -> {best_match_article} (Sim: {best_sim:.2f})")
            else:
                entry = {
                    "article": article,
                    "status": "REMOVED",
                    "summary": f"{article_label} chỉ xuất hiện trong bản A (bản cũ). Điều khoản này đã bị xóa bỏ.",
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
                }
                results.append(entry)
                if verbose:
                    print(f"  [REMOVED ] {article_label}")
            continue

        # ── Case 3: Có ở cả 2
        # Tối ưu hóa: Kiểm tra khớp tuyệt đối (Exact Match) để bỏ qua LLM
        if normalize_for_compare(text_a) == normalize_for_compare(text_b):
            if verbose:
                print(f"  [COMPARE ] {article_label} — khớp hoàn toàn (bỏ qua LLM)... IDENTICAL")
            entry = {
                "article": article,
                "status": "IDENTICAL",
                "summary": "Nội dung hai bản giống nhau hoàn toàn (Bypass LLM).",
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
            }
            results.append(entry)
        else:
            # Có thay đổi số liệu hay không?
            is_numeric = has_numeric_diff(text_a, text_b)

            # Tối ưu chặng 1: Dùng thuật toán Lexical Jaccard/Sequence (Chạy C) siêu nhẹ để dò
            lexical_sim = SequenceMatcher(None, text_a, text_b).ratio()
            
            is_minor = False
            sim = 0.0
            
            if not is_numeric and lexical_sim >= 0.93:
                sim = lexical_sim
                is_minor = True
            else:
                # Tối ưu chặng 2: Nếu đảo vị trí quá nhiều khiến thuật toán chữ bị thấp, đành dùng AI Vector ngữ nghĩa
                vec_a, norm_a = get_vec_a(article)
                vec_b, norm_b = get_vec_b(article)
                sim = np.dot(vec_a, vec_b) / (norm_a * norm_b) if (norm_a > 0 and norm_b > 0) else 0.0
                if not is_numeric and sim >= 0.93:
                    is_minor = True

            # Bypass LLM nếu không đổi số liệu VÀ độ tương đồng cao
            if is_minor:
                if verbose:
                    print(f"  [COMPARE ] {article_label} — Sim {sim:.3f} >= 0.93 & Không đổi số (bỏ qua LLM)... MINOR-MODIFIED")
                entry = {
                    "article": article,
                    "status": "MINOR-MODIFIED",
                    "summary": "Thay đổi chủ yếu ở cách diễn đạt, chưa thấy thay đổi bản chất nội dung.",
                    "exact_difference": "Chỉ thay đổi cách diễn đạt hoặc trật tự một số từ ngữ nhỏ.",
                    "change_type": "Thay đổi nhỏ về câu chữ",
                    "impact_level": "Nhỏ",
                    "citation_a": text_a[:100] + "...",
                    "citation_b": text_b[:100] + "...",
                    "text_a": text_a,
                    "text_b": text_b,
                    "word_diff": generate_word_diff(text_a, text_b),
                    "pdf_anchor_a": provenance_a,
                    "pdf_anchor_b": provenance_b,
                    "citation_anchor_a": _resolve_citation_anchor(text_a[:100], provenance_a),
                    "citation_anchor_b": _resolve_citation_anchor(text_b[:100], provenance_b),
                }
                results.append(entry)
            else:
                # Bắt buộc qua LLM (nếu có đổi số hoặc similarity thấp)
                task = {
                    "article": article,
                    "article_label": article_label,
                    "text_a": text_a,
                    "text_b": text_b,
                    "is_numeric": is_numeric,
                    "result_index": len(results),
                    "provenance_a": provenance_a,
                    "provenance_b": provenance_b,
                }
                pending_llm_tasks.append(task)
                results.append(None)  # Placeholder cho LLM xử lý sau

    # ── Chạy đa luồng cho các task cần LLM gộp lại
    if pending_llm_tasks:
        if verbose:
            print(f"\n🚀 Đang chạy đa luồng ({len(pending_llm_tasks)} điều khoản) gửi tới LLM...")

        def _run_llm(task):
            res = compare_clauses(
                task["text_a"], 
                task["text_b"], 
                task["article_label"], 
                model=model, 
                has_numeric_change=task["is_numeric"]
            )
            return task, res

        max_workers = min(2, len(pending_llm_tasks))  # Giảm xuống 2 luồng để tránh CPU bị quá tải gây timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_llm, t): t for t in pending_llm_tasks}
            
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                try:
                    _, llm_result = future.result()
                    if verbose:
                        print(f"  [COMPARE ] {task['article_label']} — LLM phân tích xong: {llm_result.get('status', 'ERROR')}")
                except Exception as e:
                    if verbose:
                        print(f"  [ERROR   ] {task['article_label']} — LLM gặp lỗi: {e}")
                    llm_result = {"status": "ERROR", "summary": f"Lỗi gọi LLM đa luồng: {e}"}

                # Cập nhật kết quả vào danh sách chính
                results[task["result_index"]] = {
                    "article": task["article"],
                    "status": llm_result.get("status", "ERROR"),
                    "summary": llm_result.get("summary", ""),
                    "exact_difference": llm_result.get("exact_difference", ""),
                    "change_type": llm_result.get("change_type", ""),
                    "citation_a": llm_result.get("citation_a", ""),
                    "citation_b": llm_result.get("citation_b", ""),
                    "text_a": task["text_a"],
                    "text_b": task["text_b"],
                    "word_diff": generate_word_diff(task["text_a"], task["text_b"]),
                    "pdf_anchor_a": task["provenance_a"],
                    "pdf_anchor_b": task["provenance_b"],
                    "citation_anchor_a": _resolve_citation_anchor(llm_result.get("citation_a", ""), task["provenance_a"]),
                    "citation_anchor_b": _resolve_citation_anchor(llm_result.get("citation_b", ""), task["provenance_b"]),
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

    return results


def save_report(results: List[dict], output_path: str = "comparison_report.json"):
    """
    Lưu kết quả so sánh ra file JSON.
    - Chỉ lưu các điều khoản CÓ thay đổi (bỏ qua IDENTICAL).
    - Loại bỏ field 'word_diff' để file gọn hơn.
    """
    SKIP_STATUSES = {"IDENTICAL"}
    SKIP_FIELDS   = {"word_diff"}

    filtered = [
        {k: v for k, v in r.items() if k not in SKIP_FIELDS}
        for r in results
        if r.get("status") not in SKIP_STATUSES
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    total   = len(results)
    saved   = len(filtered)
    skipped = total - saved
    print(f"\n✅ Đã lưu báo cáo: {output_path}")
    print(f"   → {saved} điều khoản có thay đổi được lưu ({skipped} IDENTICAL bị bỏ qua)")


def print_report(results: List[dict]):
    """
    In kết quả so sánh theo 3 phần:
      1. DANH SÁCH THAY ĐỔI  — phân loại: Thêm / Xóa / Sửa
      2. TÓM TẮT ĐIỂM QUAN TRỌNG — chỉ liệt kê những điều có thay đổi đáng chú ý
      3. TRÍCH ĐOẠN + VỊ TRÍ — evidence cụ thể cho mỗi thay đổi
    """
    SEP  = "═" * 70
    SEP2 = "─" * 70
    SEP3 = "·" * 70

    # ── Phân loại kết quả
    added   = [r for r in results if r["status"] == "ADDED"]
    removed = [r for r in results if r["status"] == "REMOVED"]
    moved   = [r for r in results if r["status"] == "MOVED"]
    modified = [r for r in results if r["status"] in ("MODIFIED", "MINOR-MODIFIED")]
    identical = [r for r in results if r["status"] == "IDENTICAL"]
    errors  = [r for r in results if r["status"] == "ERROR"]

    def article_label(r):
        a = r.get("article", "")
        return a if "Điều" in a else f"Điều {a}"

    # ══════════════════════════════════════════════════════
    # PHẦN 1 – DANH SÁCH THAY ĐỔI
    # ══════════════════════════════════════════════════════
    print("\n" + SEP)
    print("  📋  PHẦN 1 — DANH SÁCH THAY ĐỔI (THÊM / XÓA / SỬA)")
    print(SEP)

    # 1a. Điều khoản mới được thêm vào
    print(f"\n  ➕  THÊM MỚI  ({len(added)} điều khoản)")
    print(SEP2)
    if added:
        for r in added:
            print(f"    [+] {article_label(r)}")
            if r.get("change_type"):
                print(f"        Loại: {r['change_type']}")
    else:
        print("    (Không có điều khoản nào được thêm mới)")

    # 1b. Điều khoản bị xóa bỏ
    print(f"\n  ➖  XÓA BỎ  ({len(removed)} điều khoản)")
    print(SEP2)
    if removed:
        for r in removed:
            print(f"    [-] {article_label(r)}")
            if r.get("change_type"):
                print(f"        Loại: {r['change_type']}")
    else:
        print("    (Không có điều khoản nào bị xóa bỏ)")

    # 1c. Điều khoản bị di chuyển
    if moved:
        print(f"\n  🔀  DI CHUYỂN  ({len(moved)} điều khoản)")
        print(SEP2)
        for r in moved:
            print(f"    [>] {article_label(r)}")
            print(f"        {r['summary']}")

    # 1d. Điều khoản bị sửa đổi
    print(f"\n  ✏️   SỬA ĐỔI  ({len(modified)} điều khoản)")
    print(SEP2)
    if modified:
        for r in modified:
            status_tag = "[~]" if r["status"] == "MODIFIED" else "[≈]"
            print(f"    {status_tag} {article_label(r)}")
            if r.get("change_type"):
                print(f"        Loại: {r['change_type']}")
    else:
        print("    (Không có điều khoản nào bị sửa đổi)")

    # 1e. Thống kê tổng quát
    total_changed = len(added) + len(removed) + len(moved) + len(modified)
    print(f"\n  📊  Tổng số điều khoản cần xét : {len(results)}")
    print(f"      Không thay đổi              : {len(identical)}")
    print(f"      Có thay đổi                 : {total_changed}")
    if errors:
        print(f"      Lỗi xử lý                  : {len(errors)}")

    # ══════════════════════════════════════════════════════
    # PHẦN 2 – TÓM TẮT ĐIỂM THAY ĐỔI QUAN TRỌNG
    # ══════════════════════════════════════════════════════
    print("\n" + SEP)
    print("  🔍  PHẦN 2 — TÓM TẮT ĐIỂM THAY ĐỔI QUAN TRỌNG")
    print(SEP)
    print(SEP)

    important = added + removed + moved + modified
    if not important:
        print("\n  (Không phát hiện thay đổi quan trọng — toàn bộ nội dung giống nhau)")
    else:
        for idx, r in enumerate(important, 1):
            status_icon = {
                "ADDED": "➕", "REMOVED": "➖", "MOVED": "🔀",
                "MODIFIED": "✏️ ", "MINOR-MODIFIED": "≈ ",
            }.get(r["status"], "•")
            print(f"\n  {idx:>2}. {status_icon} Điều/khoản: {article_label(r)}")
            print(f"      Loại thay đổi: {r.get('change_type', r.get('status'))}")
            print(f"      Tóm tắt dễ hiểu: {r.get('summary', '(không có)')}")
            
            citation_a = (r.get("citation_a") or "").replace('\n', ' ').strip()
            citation_b = (r.get("citation_b") or "").replace('\n', ' ').strip()
            if citation_a and citation_b:
                print(f"      Minh chứng ngắn: Từ '{citation_a[:50]}...' -> '{citation_b[:50]}...'")
            elif citation_a:
                print(f"      Minh chứng ngắn: Bỏ đoạn '{citation_a[:50]}...'")
            elif citation_b:
                print(f"      Minh chứng ngắn: Thêm đoạn '{citation_b[:50]}...'")

            if r.get("impact_level"):
                print(f"      Mức độ ảnh hưởng: {r['impact_level']}")
            
            if r.get("exact_difference"):
                print(f"      Vị trí cụ thể: {r['exact_difference']}")

    # ══════════════════════════════════════════════════════
    # PHẦN 3 – TRÍCH ĐOẠN + VỊ TRÍ
    # ══════════════════════════════════════════════════════
    print("\n" + SEP)
    print("  📌  PHẦN 3 — TRÍCH ĐOẠN VÀ VỊ TRÍ CỤ THỂ")
    print(SEP)

    evidence_items = [r for r in important if r.get("citation_a") or r.get("citation_b")]
    if not evidence_items:
        print("\n  (Không có trích đoạn nào được ghi nhận)")
    else:
        for r in evidence_items:
            print(f"\n  {SEP3}")
            print(f"  📍 {article_label(r)}  [{r['status']}]")
            if r.get("citation_a"):
                print(f"\n     📄 Bản CŨ  (Doc A):")
                # Hiển thị tối đa 300 ký tự
                excerpt_a = r["citation_a"][:300] + ("…" if len(r["citation_a"]) > 300 else "")
                for line in excerpt_a.splitlines():
                    print(f"        {line}")
            if r.get("citation_b"):
                print(f"\n     📄 Bản MỚI (Doc B):")
                excerpt_b = r["citation_b"][:300] + ("…" if len(r["citation_b"]) > 300 else "")
                for line in excerpt_b.splitlines():
                    print(f"        {line}")

    print("\n" + SEP + "\n")
