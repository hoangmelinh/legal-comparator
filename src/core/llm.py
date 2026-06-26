import hashlib
import json
import os
import urllib.error
import urllib.request
from collections.abc import Generator

from src.core.cache import cache

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:7b-instruct-q3_K_M"
COMPARE_PROMPT_VERSION = "compare_v3_full"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def stream_chat_ollama(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    timeout: int = 300,
    options: dict | None = None,
) -> Generator[str, None, None]:
    """
    Gọi Ollama /api/chat với stream=True, pass danh sách messages.
    """
    merged_options = {
        "temperature": _env_float("CHAT_TEMPERATURE", 0.1),
        "top_p": _env_float("CHAT_TOP_P", 0.8),
        "num_ctx": _env_int("CHAT_NUM_CTX", 2048),
        "num_predict": _env_int("CHAT_NUM_PREDICT", 256),
    }
    if options:
        merged_options.update({key: value for key, value in options.items() if value is not None})

    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": merged_options,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_CHAT_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # /api/chat trả về data["message"]["content"]
                    token = (data.get("message") or {}).get("content", "")
                    if token:
                        yield token
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue
    except urllib.error.URLError as exc:
        raise ConnectionError(
            "Không thể kết nối Ollama. Hãy chắc chắn Ollama đang chạy.\n"
            f"Chi tiết: {exc}"
        )


def _call_ollama(prompt: str, model: str = DEFAULT_MODEL, timeout: int = 300) -> str:
    num_ctx = int(os.getenv("COMPARE_OLLAMA_NUM_CTX", "2048"))
    num_predict = int(os.getenv("COMPARE_OLLAMA_NUM_PREDICT", "384"))
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.8,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            },
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "").strip()
    except urllib.error.URLError as exc:
        raise ConnectionError(
            "Không thể kết nối Ollama. ãy chắc chắn Ollama đang chạy.\n"
            f"Chi tiết: {exc}"
        )


def _truncate(text: str, max_chars: int = 800) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _normalize_cache_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def _hash_text(text: str) -> str:
    return hashlib.sha256(_normalize_cache_text(text).encode("utf-8")).hexdigest()


def _compare_cache_key(
    model: str,
    article_label: str,
    text_a: str,
    text_b: str,
    compare_type: str,
) -> str:
    key_payload = {
        "model": model,
        "prompt_version": COMPARE_PROMPT_VERSION,
        "text_a": _hash_text(text_a),
        "text_b": _hash_text(text_b),
        "article": article_label,
        "compare_type": compare_type,
    }
    raw = json.dumps(key_payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _parse_json_response(raw: str) -> dict:
    cleaned = (raw or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def compare_clauses(
    text_a: str,
    text_b: str,
    article_label: str,
    model: str = DEFAULT_MODEL,
    has_numeric_change: bool = False,
    compare_type: str = "MODIFIED",
) -> dict:
    """
    Compare two clause texts with Ollama and return a normalized JSON payload.
    """
    timeout = int(os.getenv("COMPARE_LLM_TIMEOUT", "300"))
    text_a = _truncate(text_a)
    text_b = _truncate(text_b)

    alert_numeric = ""
    if has_numeric_change:
        alert_numeric = (
            "\nCẢNH BÁO: ĐÃ PHÁT HIỆN THAY ĐỔI VE SỐ LIỆU ĐỊNH LUỘNG.\n"
            "Trong summary, hãy chỉ rõ số cũ, số mới, hướng thay đổi và ảnh hưởng chính.\n"
        )

    cache_key = _compare_cache_key(
        model, article_label, text_a, text_b, compare_type=compare_type
    )
    cached_result = cache.get_by_key(cache_key)
    if cached_result:
        return {**cached_result, "__cache_hit": True}

    prompt = f"""Bạn là chuyên gia phân tích pháp lý. Nhiệm vụ của bạn là so sánh hai phiên bản của cùng một điều khoản hợp đồng.
ĐIỀU KHOẢN: {article_label}

--- BẢN A ---
{text_a}

--- BẢN B ---
{text_b}

Hãy phân tích và trả về DUNG JSON, không thêm bất kỳ văn bản nào ngoài JSON:

{{
  "status": "IDENTICAL hoặc MODIFIED",
  "summary": "Tóm tắt dễ hiểu (1-3 câu) về điểm khác biệt. Nếu IDENTICAL thì ghi 'Nội dung hai bản giống nhau.'",
  "important_change": true,
  "risk_note": "ếu thay đổi có thể ảnh hưởng đến quyền, nghĩa vụ, rủi ro, thời hạn, thanh toán, phạt, bồi thường thì ghi ngắn gọn. Nếu không thì để trống.",
  "exact_difference": "Chỉ rõ cụ thể phần nào thay đổi, thay đổi từ gì sang gì.",
  "change_type": "ột trong các loại: 'Thêm mới', 'óa bỏ', 'ửa nội dung', 'Di chuyển vị trí', 'Thay đổi nhỏ về câu chữ'.",
  "impact_level": "ức độ ảnh hưởng: 'Nhỏ', 'Trung bình', 'ớn', hoặc để trống nếu IDENTICAL.",
  "citation_a": "Trích nguyên văn phần bị đổi từ BẢN A. Tối đa 50 từ.",
  "citation_b": "Trích nguyên văn phần mới từ BẢN B. Tối đa 50 từ."
}}

QUY TẮC:
1. Không dùng nhận xét mơ hồ.
2. Nếu có thay đổi, phải chỉ rõ thay đổi từ gì sang gì và ảnh hưởng chính.
3. Không được sáng tác citation; citation phải là trích đoạn có trong BẢN A/B.
4. Không trả về markdown, chỉ trả về JSON.
5. Đánh dấu important_change=true nếu có thay đổi về số tiền, phần trăm, ngày tháng, thời hạn, chủ thể, quyền, nghĩa vụ, thanh toán, phạt, bồi thường, bảo mật, chậm chạp hoặc điều kiện đơn phương.
{alert_numeric}
"""

    last_raw = ""
    for attempt in range(2):
        raw = _call_ollama(prompt, model=model, timeout=timeout)
        last_raw = raw
        try:
            result = _parse_json_response(raw)
            for key in (
                "status",
                "summary",
                "exact_difference",
                "change_type",
                "impact_level",
                "citation_a",
                "citation_b",
                "risk_note",
            ):
                result.setdefault(key, "")
            result.setdefault("important_change", bool(has_numeric_change))
            result["status"] = "MODIFIED"
            cache.set_by_key(cache_key, result)
            return {**result, "__cache_hit": False}
        except json.JSONDecodeError:
            if attempt == 0:
                prompt += "\nSửa lỗi: lần trước không phải JSON hợp lệ. Chỉ trả về một object JSON."
                continue

    return {
        "status": "ERROR",
        "summary": f"LLM trả về định dạng không đúng JSON. Raw: {last_raw[:200]}",
        "exact_difference": "",
        "change_type": "",
        "impact_level": "",
        "citation_a": "",
        "citation_b": "",
        "__cache_hit": False,
    }


def is_ollama_running(model: str = DEFAULT_MODEL) -> bool:
    try:
        _call_ollama("Hi", model=model, timeout=60)
        return True
    except (ConnectionError, Exception):
        return False
