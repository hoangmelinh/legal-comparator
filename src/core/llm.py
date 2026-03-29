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
                    model: str = DEFAULT_MODEL) -> dict:
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
    prompt = f"""Bạn là chuyên gia phân tích pháp lý. Nhiệm vụ của bạn là so sánh hai phiên bản của cùng một điều khoản hợp đồng.

ĐIỀU KHOẢN: {article_label}

--- BẢN A ---
{text_a}

--- BẢN B ---
{text_b}

Hãy phân tích và trả về KẾT QUẢ THEO ĐÚNG ĐỊNH DẠNG JSON dưới đây, KHÔNG thêm bất kỳ văn bản nào ngoài JSON:

{{
  "status": "IDENTICAL hoặc MODIFIED",
  "summary": "Tóm tắt ngắn gọn (1-3 câu) về điểm khác biệt. Nếu IDENTICAL thì ghi 'Nội dung hai bản giống nhau.'",
  "exact_difference": "Chỉ rõ cụ thể điểm nào/khoản nào bị thay đổi (ví dụ: 'Khoản 2', 'Điểm a Khoản 3') hoặc 'Toàn bộ điều khoản'. Nếu IDENTICAL thì để chuỗi rỗng.",
  "change_type": "Hãy ghi rõ 'Thay đổi từ ngữ' (nếu chỉ đổi cách diễn đạt) HOẶC 'Thay đổi ý nghĩa' (nếu có khác biệt về quyền lợi/nghĩa vụ pháp lý). Nếu IDENTICAL thì để chuỗi rỗng.",
  "citation_a": "Trích nguyên văn đoạn quan trọng nhất từ BẢN A làm bằng chứng (tối đa 100 từ). Nếu IDENTICAL thì để chuỗi rỗng.",
  "citation_b": "Trích nguyên văn đoạn quan trọng nhất từ BẢN B làm bằng chứng (tối đa 100 từ). Nếu IDENTICAL thì để chuỗi rỗng."
}}

QUY TẮC BẮT BUỘC:
- Chỉ kết luận MODIFIED nếu có sự khác biệt thực sự về nội dung (không phải về định dạng/khoảng trắng).
- Phần citation PHẢI là trích dẫn nguyên văn từ văn bản, KHÔNG được tự sáng tác.
- Trả về JSON hợp lệ, không có markdown code block.
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
        for key in ("status", "summary", "exact_difference", "change_type", "citation_a", "citation_b"):
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
