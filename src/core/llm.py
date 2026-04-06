import json
import urllib.request
import urllib.error

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:7b"


def _call_ollama(prompt: str, model: str = DEFAULT_MODEL, timeout: int = 300) -> str:
    """
    Gọi Ollama API và trả về nội dung response dạng string.
    """
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,   # Can ket qua on dinh, khong ngau nhien
            "num_ctx": 2048,      # Giam de chay nhanh hon tren CPU
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "").strip()
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Không thể kết nối Ollama. Hãy chắc chắn Ollama đang chạy.\nChi tiết: {e}"
        )


from src.core.cache import cache

def compare_clauses(text_a: str, text_b: str, article_label: str,
                    model: str = DEFAULT_MODEL, has_numeric_change: bool = False) -> dict:
    """
    Dùng LLM so sánh 2 điều khoản và trả về kết quả JSON có cấu trúc.

    Returns:
    {
        "status": "MODIFIED" | "IDENTICAL" | "ERROR",
        "summary": "Tóm tắt điểm khác biệt...",
        "citation_a": "Đoạn trích từ bản A làm bằng chứng...",
        "citation_b": "Đoạn trích từ bản B làm bằng chứng..."
    }
    """
    alert_numeric = ""
    if has_numeric_change:
        alert_numeric = "\n⚡ CẢNH BÁO: ĐÃ PHÁT HIỆN THAY ĐỔI VỀ MẶT SỐ LIỆU ĐỊNH LƯỢNG (Tiền, tỷ lệ, thời gian...).\n" \
                        "BẠN TẬP TRUNG PHÂN TÍCH VÀ TRẢ LỜI 3 CÂU HỎI SAU TRONG PHẦN 'summary':\n" \
                        "- Số cũ là bao nhiêu? Số mới là bao nhiêu?\n" \
                        "- Thay đổi này theo hướng nào (tăng/giảm, kéo dài/rút ngắn...)?\n" \
                        "- Ảnh hưởng chính của việc thay đổi số liệu này tới bản chất hợp đồng là gì?\n"

    prompt = f"""Bạn là chuyên gia phân tích pháp lý. Nhiệm vụ của bạn là so sánh hai phiên bản của cùng một điều khoản hợp đồng.

ĐIỀU KHOẢN: {article_label}

--- BẢN A ---
{text_a}

--- BẢN B ---
{text_b}

Hãy phân tích và trả về KẾT QUẢ THEO ĐÚNG ĐỊNH DẠNG JSON dưới đây, KHÔNG thêm bất kỳ văn bản nào ngoài JSON:

{{
  "status": "IDENTICAL hoặc MODIFIED",
  "summary": "Tóm tắt dễ hiểu (1-3 câu) về điểm khác biệt. Xem QUY TẮC TÓM TẮT DƯỚI ĐÂY.",
  "exact_difference": "Chỉ rõ cụ thể phần nào thay đổi (Ví dụ: 'Khoản 2', 'Điểm a Khoản 3', 'Toàn bộ điều khoản').",
  "change_type": "Một trong các loại: 'Thêm mới', 'Xóa bỏ', 'Sửa nội dung', 'Di chuyển vị trí', 'Thay đổi nhỏ về câu chữ'.",
  "impact_level": "Mức độ ảnh hưởng: 'Nhỏ', 'Trung bình', 'Lớn', hoặc để trống nếu IDENTICAL.",
  "citation_a": "Trích nguyên văn phần bị đổi từ BẢN A làm minh chứng. Tối đa 50 từ.",
  "citation_b": "Trích nguyên văn phần mới từ BẢN B làm minh chứng. Tối đa 50 từ."
}}

QUY TẮC TÓM TẮT CHO PHẦN "summary":
1. Dùng ngôn ngữ tự nhiên, chính xác, dễ hiểu. Nếu IDENTICAL thì ghi 'Nội dung hai bản giống nhau.'
2. TUYỆT ĐỐI KHÔNG dùng các câu mơ hồ như: 'có thay đổi nội dung', 'điều khoản được chỉnh sửa', 'nội dung có khác biệt'.
3. Phải chỉ ra cụ thể: phần nào thay đổi, thay đổi từ gì sang gì, theo hướng nào và ảnh hưởng chính là gì.
4. Ưu tiên mẫu câu: 'Trước đây [nội dung/quy định cũ], nay sửa thành [nội dung/quy định mới]' hoặc 'Điều khoản này thay đổi từ [cũ] sang [mới]'. Căn cứ vào đó bản mới bổ sung/bỏ/điều chỉnh...
5. Lấy thay đổi lớn đưa lên đầu. Ưu tiên nêu: quyền nào, nghĩa vụ nào, điều kiện nào, thời hạn, mức tiền, hay chế tài nào đã thay đổi.
6. Nếu nội dung thay đổi nhỏ (về câu chữ/diễn đạt), phải ghi rõ thay đổi đó và kết luận có đổi ý nghĩa hay không.
7. Nếu không đủ dữ liệu/ngữ cảnh để kết luận, TUYỆT ĐỐI KHÔNG tự suy diễn, BẮT BUỘC thêm: 'Chưa đủ dữ liệu để kết luận chính xác' hoặc 'Cần xem thêm ngữ cảnh của điều khoản'.
8. KHÔNG liệt kê diff theo từng từ. Không lặp lại nguyên văn quá nhiều (dành việc đó cho citation). Viết để người bình thường đọc hiểu ngay sự khác biệt.
{alert_numeric}
QUY TẮC BẮT BUỘC KHÁC:
- Chỉ kết luận MODIFIED nếu có sự thay đổi. Định dạng khoảng trắng/newline nếu đọc tương đồng thì là IDENTICAL.
- Phần citation PHẢI lấy chính xác từ BẢN A và BẢN B. KHÔNG SÁNG TÁC.
- TRẢ VỀ CHỈ JSON, KHÔNG text ngoài, KHÔNG markdown block.
"""

    # Truncate input to avoid timeout on CPU (max ~800 chars each)
    MAX_CHARS = 800
    if len(text_a) > MAX_CHARS:
        text_a = text_a[:MAX_CHARS] + "..."
    if len(text_b) > MAX_CHARS:
        text_b = text_b[:MAX_CHARS] + "..."

    # Check cache first
    cached_result = cache.get(prompt, model)
    if cached_result:
        return cached_result

    raw = _call_ollama(prompt, model=model)

    # Parse JSON từ response
    try:
        # Đôi khi LLM bọc trong ```json ... ```, cần strip ra
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

        result = json.loads(cleaned)

        # Validate keys
        for key in ("status", "summary", "exact_difference", "change_type", "impact_level", "citation_a", "citation_b"):
            if key not in result:
                result[key] = ""

        # Update cache
        cache.set(prompt, model, result)
        
        return result

    except json.JSONDecodeError:
        return {
            "status": "ERROR",
            "summary": f"LLM trả về định dạng không đúng JSON. Raw: {raw[:200]}",
            "exact_difference": "",
            "change_type": "",
            "impact_level": "",
            "citation_a": "",
            "citation_b": ""
        }


def is_ollama_running(model: str = DEFAULT_MODEL) -> bool:
    """Kiểm tra Ollama API có đang chạy không."""
    try:
        _call_ollama("Hi", model=model, timeout=60)
        return True
    except (ConnectionError, Exception):
        return False
