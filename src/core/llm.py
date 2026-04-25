import json
import urllib.error
import urllib.request
from collections.abc import Generator

from src.core.cache import cache

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:3b"


def stream_chat_ollama(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    timeout: int = 300,
) -> Generator[str, None, None]:
    """
    Gọi Ollama /api/chat với stream=True, pass danh sách messages.
    """
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0.2,
                "num_ctx": 2048,
                "num_predict": 400,
            },
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
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": 2048,
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
            "Khong the ket noi Ollama. Hay chac chan Ollama dang chay.\n"
            f"Chi tiet: {exc}"
        )


def _truncate(text: str, max_chars: int = 800) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


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
) -> dict:
    """
    Compare two clause texts with Ollama and return a normalized JSON payload.
    """
    text_a = _truncate(text_a)
    text_b = _truncate(text_b)

    alert_numeric = ""
    if has_numeric_change:
        alert_numeric = (
            "\nCANH BAO: DA PHAT HIEN THAY DOI VE SO LIEU DINH LUONG.\n"
            "Trong summary, hay chi ro so cu, so moi, huong thay doi va anh huong chinh.\n"
        )

    prompt = f"""Ban la chuyen gia phan tich phap ly. Nhiem vu cua ban la so sanh hai phien ban cua cung mot dieu khoan hop dong.

DIEU KHOAN: {article_label}

--- BAN A ---
{text_a}

--- BAN B ---
{text_b}

Hay phan tich va tra ve DUNG JSON, khong them bat ky van ban nao ngoai JSON:

{{
  "status": "IDENTICAL hoac MODIFIED",
  "summary": "Tom tat de hieu (1-3 cau) ve diem khac biet. Neu IDENTICAL thi ghi 'Noi dung hai ban giong nhau.'",
  "exact_difference": "Chi ro cu the phan nao thay doi.",
  "change_type": "Mot trong cac loai: 'Them moi', 'Xoa bo', 'Sua noi dung', 'Di chuyen vi tri', 'Thay doi nho ve cau chu'.",
  "impact_level": "Muc do anh huong: 'Nho', 'Trung binh', 'Lon', hoac de trong neu IDENTICAL.",
  "citation_a": "Trich nguyen van phan bi doi tu BAN A. Toi da 50 tu.",
  "citation_b": "Trich nguyen van phan moi tu BAN B. Toi da 50 tu."
}}

QUY TAC:
1. Khong dung nhan xet mo ho.
2. Neu co thay doi, phai chi ro thay doi tu gi sang gi va anh huong chinh.
3. Khong duoc sang tac citation.
4. Khong tra ve markdown, chi tra ve JSON.
{alert_numeric}
"""

    cached_result = cache.get(prompt, model)
    if cached_result:
        return cached_result

    raw = _call_ollama(prompt, model=model)
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
        ):
            result.setdefault(key, "")
        cache.set(prompt, model, result)
        return result
    except json.JSONDecodeError:
        return {
            "status": "ERROR",
            "summary": f"LLM tra ve dinh dang khong dung JSON. Raw: {raw[:200]}",
            "exact_difference": "",
            "change_type": "",
            "impact_level": "",
            "citation_a": "",
            "citation_b": "",
        }


def is_ollama_running(model: str = DEFAULT_MODEL) -> bool:
    try:
        _call_ollama("Hi", model=model, timeout=60)
        return True
    except (ConnectionError, Exception):
        return False
