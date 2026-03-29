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

    for article in all_articles:
        in_a = article in grouped_a
        in_b = article in grouped_b

        text_a = _merge_chunks_text(grouped_a[article]) if in_a else ""
        text_b = _merge_chunks_text(grouped_b[article]) if in_b else ""
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
            }
            results.append(entry)
            if verbose:
                print(f"  [ADDED   ] {article_label}")
            continue

        # ── Case 2: Chỉ có ở bản A → ĐÃ BỊ XÓA HOẶC DI CHUYỂN
        if in_a and not in_b:
            vec_a = db.embedder.get_embeddings([text_a])[0]
            norm_a = np.linalg.norm(vec_a)
            
            best_match_article = None
            best_sim = 0.0
            best_text_b = ""
            
            for art_b, chunks_b_list in grouped_b.items():
                txt_b = _merge_chunks_text(chunks_b_list)
                if not txt_b.strip(): continue
                vec_b = db.embedder.get_embeddings([txt_b])[0]
                norm_b = np.linalg.norm(vec_b)
                sim = np.dot(vec_a, vec_b) / (norm_a * norm_b) if (norm_a > 0 and norm_b > 0) else 0.0
                
                if sim > best_sim:
                    best_sim = sim
                    best_match_article = art_b if "Điều" in art_b else f"Điều {art_b}"
                    best_text_b = txt_b
            
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
                    "word_diff": generate_word_diff(text_a, best_text_b)
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
                }
                results.append(entry)
                if verbose:
                    print(f"  [REMOVED ] {article_label}")
            continue

        # ── Case 3: Có ở cả 2
        # Tối ưu hóa: Kiểm tra khớp tuyệt đối (Exact Match) để bỏ qua LLM
        def normalize_for_compare(t):
            import unicodedata
            t = unicodedata.normalize('NFC', t)
            return re.sub(r'\s+', ' ', t).strip().lower()
            
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
            }
            results.append(entry)
        else:
            # Cosine similarity check bypass
            vec_a = db.embedder.get_embeddings([text_a])[0]
            vec_b = db.embedder.get_embeddings([text_b])[0]
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            sim = np.dot(vec_a, vec_b) / (norm_a * norm_b) if (norm_a > 0 and norm_b > 0) else 0.0

            if sim >= 0.98:
                if verbose:
                    print(f"  [COMPARE ] {article_label} — Cosine Sim {sim:.3f} >= 0.98 (bỏ qua LLM)... MINOR-MODIFIED")
                entry = {
                    "article": article,
                    "status": "MINOR-MODIFIED",
                    "summary": f"Nội dung thay đổi rất ít (Cosine Similarity: {sim:.3f}). Đã bỏ qua LLM.",
                    "exact_difference": "Chỉ thay đổi định dạng hoặc vài từ ngữ nhỏ.",
                    "change_type": "Thay đổi từ ngữ",
                    "citation_a": text_a[:100] + "...",
                    "citation_b": text_b[:100] + "...",
                    "text_a": text_a,
                    "text_b": text_b,
                    "word_diff": generate_word_diff(text_a, text_b)
                }
                results.append(entry)
            else:
                # Cần kiểm tra bằng LLM
                task = {
                    "article": article,
                    "article_label": article_label,
                    "text_a": text_a,
                    "text_b": text_b,
                    "result_index": len(results)
                }
                pending_llm_tasks.append(task)
                results.append(None)  # Placeholder cho LLM xử lý sau

    # ── Chạy đa luồng cho các task cần LLM gộp lại
    if pending_llm_tasks:
        if verbose:
            print(f"\n🚀 Đang chạy đa luồng ({len(pending_llm_tasks)} điều khoản) gửi tới LLM...")

        def _run_llm(task):
            res = compare_clauses(task["text_a"], task["text_b"], task["article_label"], model=model)
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
                    "word_diff": generate_word_diff(task["text_a"], task["text_b"])
                }

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

    important = added + removed + moved + modified
    if not important:
        print("\n  (Không phát hiện thay đổi quan trọng — toàn bộ nội dung giống nhau)")
    else:
        for idx, r in enumerate(important, 1):
            status_icon = {
                "ADDED": "➕", "REMOVED": "➖", "MOVED": "🔀",
                "MODIFIED": "✏️ ", "MINOR-MODIFIED": "≈ ",
            }.get(r["status"], "•")
            print(f"\n  {idx:>2}. {status_icon} {article_label(r)}")
            print(f"      Tóm tắt : {r.get('summary', '(không có)')}")
            if r.get("exact_difference"):
                print(f"      Vị trí  : {r['exact_difference']}")
            if r.get("change_type"):
                print(f"      Tính chất: {r['change_type']}")

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
