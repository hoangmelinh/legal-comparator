from __future__ import annotations

import shutil
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.core.comparator import run_comparison
from src.core.document_registry import DEFAULT_CACHE_ROOT
from src.core.ingestion import ingest_document, prepare_document
from src.core.llm import DEFAULT_MODEL, stream_chat_ollama
from src.database.vector_store import LegalVectorDB

DB_PATH = "legal_data.json"
CACHE_ROOT = Path(DEFAULT_CACHE_ROOT)
UPLOAD_ROOT = CACHE_ROOT / "uploads"
COMPARE_CACHE_PATH = CACHE_ROOT / "compare_cache.json"
DOCUMENT_REGISTRY_PATH = CACHE_ROOT / "document_registry.json"

for directory in (UPLOAD_ROOT, CACHE_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Legal Comparator Backend", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _safe_unlink(path_value: Any) -> None:
    if not path_value:
        return
    path = Path(path_value)
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _safe_clear_directory(directory: Path) -> None:
    if not directory.exists():
        return
    for child in directory.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except Exception:
            pass


def _reset_generated_artifacts() -> None:
    for directory in (
        UPLOAD_ROOT,
        CACHE_ROOT / "normalized_pdf",
        CACHE_ROOT / "extracted_text",
        CACHE_ROOT / "pdf_records",
        CACHE_ROOT / "vector_cache",
        CACHE_ROOT / "chunk_cache",
    ):
        _safe_clear_directory(directory)
    _safe_unlink(COMPARE_CACHE_PATH)
    _safe_unlink(DOCUMENT_REGISTRY_PATH)


class JobState(BaseModel):
    job_id: str
    job_type: Literal["process", "compare"]
    workflow_id: str
    current_phase: Literal["file_1", "file_2", "compare"]
    progress_percent: float = 0
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    message: str = ""
    started_at: str
    finished_at: str | None = None
    error: str | None = None
    document_id: str | None = None
    compare_job_id: str | None = None


class CompareRequest(BaseModel):
    workflow_id: str = "default"
    document_id_a: str | None = None
    document_id_b: str | None = None
    model: str = DEFAULT_MODEL


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    workflow_id: str = "default"
    compare_job_id: str | None = None
    message: str
    history: list[ChatMessage] = []
    model: str = DEFAULT_MODEL


state_lock = threading.Lock()
jobs: dict[str, JobState] = {}
workflow_state: dict[str, dict[str, Any]] = {}
compare_results: dict[str, dict[str, Any]] = {}


def _get_workflow(workflow_id: str) -> dict[str, Any]:
    with state_lock:
        workflow = workflow_state.get(workflow_id)
        if not workflow:
            workflow = {
                "workflow_id": workflow_id,
                "file_1": None,
                "file_2": None,
                "active_job_id": None,
                "last_compare_job_id": None,
                "created_at": _utc_now(),
            }
            workflow_state[workflow_id] = workflow
        return workflow


def _set_job_progress(job_id: str, percent: float, message: str) -> None:
    with state_lock:
        job = jobs.get(job_id)
        if not job:
            return
        job.progress_percent = max(0, min(100, round(percent, 2)))
        if message:
            job.message = message
        jobs[job_id] = job


def _build_pdf_url(document_id: str | None, workflow_id: str) -> str:
    if not document_id:
        return ""
    return f"/api/documents/{document_id}/pdf?workflow_id={workflow_id}"


def _is_reportable_change(change: dict[str, Any]) -> bool:
    status = (change.get("status") or "").upper()
    return status in {"ADDED", "REMOVED", "MOVED", "MODIFIED"}


def _normalize_change_payload(change: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(change)
    evidence = normalized.get("evidence") or {}
    citation_a = normalized.get("citation_a") or evidence.get("citation_a") or ""
    citation_b = normalized.get("citation_b") or evidence.get("citation_b") or ""
    normalized["evidence"] = {"citation_a": citation_a, "citation_b": citation_b}
    normalized.setdefault("highlight_anchors_a", [])
    normalized.setdefault("highlight_anchors_b", [])
    normalized.setdefault("changed_spans_a", [])
    normalized.setdefault("changed_spans_b", [])
    return normalized


def _clear_workflow_jobs(workflow_id: str) -> None:
    with state_lock:
        remove_ids = [
            job_id for job_id, job in jobs.items() if job.workflow_id == workflow_id
        ]
        for job_id in remove_ids:
            jobs.pop(job_id, None)


def _cleanup_document_artifacts(doc: dict[str, Any] | None) -> None:
    if not doc:
        return
    for key in (
        "original_file_path",
        "normalized_pdf_path",
        "preview_pdf_path",
        "text_cache_path",
        "records_cache_path",
        "vector_cache_path",
        "chunk_cache_path",
    ):
        _safe_unlink(doc.get(key))


def _delete_workflow_docs_from_db(workflow: dict[str, Any]) -> None:
    doc_ids = {
        doc.get("document_id")
        for doc in (workflow.get("file_1"), workflow.get("file_2"))
        if doc and doc.get("document_id")
    }
    if not doc_ids:
        return
    try:
        LegalVectorDB(db_name=DB_PATH).delete_doc_ids(doc_ids)
    except Exception:
        pass


def _reset_workflow_state(workflow_id: str, keep_current_files: bool = False) -> None:
    with state_lock:
        workflow = workflow_state.get(workflow_id) or {
            "workflow_id": workflow_id,
            "file_1": None,
            "file_2": None,
            "active_job_id": None,
            "last_compare_job_id": None,
            "created_at": _utc_now(),
        }
        previous_workflow = {
            "file_1": workflow.get("file_1"),
            "file_2": workflow.get("file_2"),
            "last_compare_job_id": workflow.get("last_compare_job_id"),
        }
        workflow_state[workflow_id] = {
            "workflow_id": workflow_id,
            "file_1": workflow.get("file_1") if keep_current_files else None,
            "file_2": workflow.get("file_2") if keep_current_files else None,
            "active_job_id": None,
            "last_compare_job_id": None,
            "created_at": _utc_now(),
        }
    if previous_workflow["last_compare_job_id"]:
        compare_results.pop(previous_workflow["last_compare_job_id"], None)
    _delete_workflow_docs_from_db(previous_workflow)
    _cleanup_document_artifacts(previous_workflow["file_1"])
    _cleanup_document_artifacts(previous_workflow["file_2"])
    _clear_workflow_jobs(workflow_id)


def _reset_slot_state(workflow_id: str, slot: Literal["file_1", "file_2"]) -> None:
    with state_lock:
        workflow = workflow_state.get(workflow_id)
        if not workflow:
            return
        previous_doc = workflow.get(slot)
        previous_compare_job = workflow.get("last_compare_job_id")
        workflow[slot] = None
        workflow["last_compare_job_id"] = None
    if previous_compare_job:
        compare_results.pop(previous_compare_job, None)
    _cleanup_document_artifacts(previous_doc)
    if previous_doc and previous_doc.get("document_id"):
        try:
            LegalVectorDB(db_name=DB_PATH).delete_doc_ids({previous_doc["document_id"]})
        except Exception:
            pass


async def _save_upload_file(upload_file: UploadFile, suffix: str) -> Path:
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(
        delete=False, dir=UPLOAD_ROOT, suffix=suffix
    )
    temp_path = Path(temp_file.name)
    try:
        with temp_file:
            while True:
                chunk = await upload_file.read(1024 * 1024)
                if not chunk:
                    break
                temp_file.write(chunk)
    except Exception:
        _safe_unlink(temp_path)
        raise
    return temp_path


def _prepare_job_payload(
    change_list: list[dict],
    workflow_id: str,
    doc_a: dict[str, Any],
    doc_b: dict[str, Any],
    compare_job_id: str,
) -> dict[str, Any]:
    normalized_changes = []
    for idx, item in enumerate(change_list, start=1):
        if not item or item.get("status") == "IDENTICAL":
            continue
        citation_a = item.get("citation_a", "")
        citation_b = item.get("citation_b", "")
        highlight_anchors_a = item.get("highlight_anchors_a", [])
        highlight_anchors_b = item.get("highlight_anchors_b", [])
        page_a = (item.get("citation_anchor_a") or {}).get("page_start") or (
            item.get("pdf_anchor_a") or {}
        ).get("page_start")
        page_b = (item.get("citation_anchor_b") or {}).get("page_start") or (
            item.get("pdf_anchor_b") or {}
        ).get("page_start")
        if not page_a and highlight_anchors_a:
            page_a = highlight_anchors_a[0].get("page") or highlight_anchors_a[0].get(
                "page_start"
            )
        if not page_b and highlight_anchors_b:
            page_b = highlight_anchors_b[0].get("page") or highlight_anchors_b[0].get(
                "page_start"
            )
        normalized_changes.append(
            {
                "id": f"chg_{idx}",
                "article": item.get("article"),
                "status": item.get("status"),
                "change_type": item.get("change_type"),
                "highlight_mode": item.get("highlight_mode", "inline"),
                "summary": item.get("summary"),
                "evidence": {"citation_a": citation_a, "citation_b": citation_b},
                "page_a": page_a,
                "page_b": page_b,
                "page": None,
                "text_a": item.get("text_a"),
                "text_b": item.get("text_b"),
                "highlight_anchors_a": highlight_anchors_a,
                "highlight_anchors_b": highlight_anchors_b,
                "changed_spans_a": item.get("changed_spans_a", []),
                "changed_spans_b": item.get("changed_spans_b", []),
                "raw": item,
            }
        )

    visible_changes = []
    for visible_index, change in enumerate(
        (change for change in normalized_changes if _is_reportable_change(change)),
        start=1,
    ):
        visible_change = dict(change)
        visible_change["id"] = f"chg_{visible_index}"
        visible_changes.append(visible_change)

    payload = {
        "compare_job_id": compare_job_id,
        "workflow_id": workflow_id,
        "document_a": {
            "document_id": doc_a.get("document_id"),
            "workflow_doc_id": doc_a.get("workflow_doc_id"),
            "pdf_url": _build_pdf_url(doc_a.get("document_id"), workflow_id),
            "file_hash": doc_a.get("file_hash"),
            "warning": doc_a.get("warning_message"),
        },
        "document_b": {
            "document_id": doc_b.get("document_id"),
            "workflow_doc_id": doc_b.get("workflow_doc_id"),
            "pdf_url": _build_pdf_url(doc_b.get("document_id"), workflow_id),
            "file_hash": doc_b.get("file_hash"),
            "warning": doc_b.get("warning_message"),
        },
        "document_a_pdf_url": _build_pdf_url(doc_a.get("document_id"), workflow_id),
        "document_b_pdf_url": _build_pdf_url(doc_b.get("document_id"), workflow_id),
        "changes": visible_changes,
        "report_changes": visible_changes,
        "report": {
            "total": len(visible_changes),
            "total_clauses": len(visible_changes),
            "added": len([c for c in visible_changes if c.get("status") == "ADDED"]),
            "removed": len(
                [c for c in visible_changes if c.get("status") == "REMOVED"]
            ),
            "modified": len(
                [c for c in visible_changes if c.get("status") == "MODIFIED"]
            ),
            "minor_modified": 0,
            "moved": len([c for c in visible_changes if c.get("status") == "MOVED"]),
        },
        "generated_at": _utc_now(),
    }
    return payload


def _start_process_job(
    workflow_id: str,
    slot: Literal["file_1", "file_2"],
    source_path: Path,
    document_id: str,
) -> str:
    job_id = str(uuid.uuid4())
    with state_lock:
        jobs[job_id] = JobState(
            job_id=job_id,
            job_type="process",
            workflow_id=workflow_id,
            current_phase=slot,
            progress_percent=0,
            status="queued",
            message="Queued",
            started_at=_utc_now(),
            document_id=document_id,
        )
        workflow_state[workflow_id]["active_job_id"] = job_id

    def _run() -> None:
        try:
            with state_lock:
                jobs[job_id].status = "running"
                jobs[job_id].message = "Processing document"

            def progress_cb(pct: float, msg: str) -> None:
                _set_job_progress(job_id, pct, msg)

            prepared = prepare_document(
                file_path=str(source_path),
                cache_root=CACHE_ROOT,
                cache_key=document_id,
                progress_callback=lambda p, m: progress_cb(max(0, min(50, p / 2)), m),
            )

            result = ingest_document(
                file_path=str(source_path),
                doc_id=document_id,
                db_path=DB_PATH,
                cache_root=CACHE_ROOT,
                cache_key=document_id,
                progress_callback=lambda p, m: progress_cb(
                    50 + max(0, min(50, p / 2)), m
                ),
            )

            with state_lock:
                workflow = workflow_state[workflow_id]
                workflow[slot] = {
                    "slot": slot,
                    "document_id": document_id,
                    "workflow_doc_id": document_id,
                    "file_hash": result.get("source_hash"),
                    "original_file_path": str(source_path),
                    "normalized_pdf_path": result.get("normalized_pdf_path"),
                    "preview_pdf_path": result.get("preview_pdf_path"),
                    "convert_status": result.get("conversion_method"),
                    "ingest_status": result.get("ingest_status"),
                    "warning_message": result.get("warning"),
                    "metadata": result,
                    "prepared_metadata": prepared,
                    "updated_at": _utc_now(),
                }
                workflow["active_job_id"] = None
                jobs[job_id].status = "completed"
                jobs[job_id].progress_percent = 100
                jobs[job_id].message = "Processing complete"
                jobs[job_id].finished_at = _utc_now()
        except Exception as exc:
            with state_lock:
                workflow_state[workflow_id]["active_job_id"] = None
                jobs[job_id].status = "failed"
                jobs[job_id].error = str(exc)
                jobs[job_id].message = "Processing failed"
                jobs[job_id].finished_at = _utc_now()
        finally:
            _safe_unlink(source_path)

    threading.Thread(target=_run, daemon=True).start()
    return job_id


def _start_compare_job(
    workflow_id: str, doc_a: dict[str, Any], doc_b: dict[str, Any], model: str
) -> str:
    job_id = str(uuid.uuid4())
    with state_lock:
        jobs[job_id] = JobState(
            job_id=job_id,
            job_type="compare",
            workflow_id=workflow_id,
            current_phase="compare",
            progress_percent=0,
            status="queued",
            message="Queued",
            started_at=_utc_now(),
            compare_job_id=job_id,
        )
        workflow_state[workflow_id]["active_job_id"] = job_id
        workflow_state[workflow_id]["last_compare_job_id"] = job_id

    def _run() -> None:
        try:
            with state_lock:
                jobs[job_id].status = "running"
                jobs[job_id].message = "Comparing documents"

            def progress_cb(pct: float, msg: str) -> None:
                _set_job_progress(job_id, pct, msg)

            results = run_comparison(
                doc_id_a=doc_a["document_id"],
                doc_id_b=doc_b["document_id"],
                db_path=DB_PATH,
                model=model,
                verbose=False,
                progress_callback=progress_cb,
            )

            payload = _prepare_job_payload(results, workflow_id, doc_a, doc_b, job_id)
            compare_results[job_id] = payload

            with state_lock:
                jobs[job_id].progress_percent = 100
                jobs[job_id].status = "completed"
                jobs[job_id].message = "Compare complete"
                jobs[job_id].finished_at = _utc_now()
                workflow_state[workflow_id]["active_job_id"] = None
        except Exception as exc:
            with state_lock:
                jobs[job_id].status = "failed"
                jobs[job_id].error = str(exc)
                jobs[job_id].message = "Compare failed"
                jobs[job_id].finished_at = _utc_now()
                workflow_state[workflow_id]["active_job_id"] = None

    threading.Thread(target=_run, daemon=True).start()
    return job_id


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    slot: Literal["file_1", "file_2"] = Query(...),
    workflow_id: str = Query("default"),
) -> dict[str, Any]:
    workflow = _get_workflow(workflow_id)

    with state_lock:
        if workflow.get("active_job_id"):
            raise HTTPException(status_code=409, detail="Another job is running")
        if slot == "file_2" and not workflow.get("file_1"):
            raise HTTPException(status_code=409, detail="Upload file_1 first")

    filename = file.filename or "uploaded.bin"
    ext = Path(filename).suffix.lower()
    if ext not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    if slot == "file_1":
        _reset_workflow_state(workflow_id, keep_current_files=False)
        _reset_generated_artifacts()
    else:
        _reset_slot_state(workflow_id, slot)

    save_path = await _save_upload_file(file, ext)
    if not save_path.exists() or save_path.stat().st_size == 0:
        _safe_unlink(save_path)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    document_id = f"{workflow_id}_{slot}_{uuid.uuid4().hex[:8]}"
    job_id = _start_process_job(
        workflow_id=workflow_id,
        slot=slot,
        source_path=save_path,
        document_id=document_id,
    )

    return {
        "workflow_id": workflow_id,
        "slot": slot,
        "document_id": document_id,
        "job_id": job_id,
        "status": "accepted",
        "message": "Upload accepted",
    }


@app.get("/api/progress/{job_id}")
def get_job_progress(job_id: str) -> dict[str, Any]:
    with state_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job.model_dump()


@app.post("/api/compare")
def compare_documents(payload: CompareRequest) -> dict[str, Any]:
    workflow = _get_workflow(payload.workflow_id)

    with state_lock:
        if workflow.get("active_job_id"):
            raise HTTPException(status_code=409, detail="Another job is running")

        doc_a = workflow.get("file_1") if not payload.document_id_a else None
        doc_b = workflow.get("file_2") if not payload.document_id_b else None

        if payload.document_id_a:
            doc_a = next(
                (
                    candidate
                    for candidate in (workflow.get("file_1"), workflow.get("file_2"))
                    if candidate
                    and candidate.get("document_id") == payload.document_id_a
                ),
                None,
            )
        if payload.document_id_b:
            doc_b = next(
                (
                    candidate
                    for candidate in (workflow.get("file_1"), workflow.get("file_2"))
                    if candidate
                    and candidate.get("document_id") == payload.document_id_b
                ),
                None,
            )

        if not doc_a or not doc_b:
            raise HTTPException(
                status_code=400, detail="File_1 and file_2 must be processed first"
            )

    compare_job_id = _start_compare_job(
        payload.workflow_id, doc_a, doc_b, payload.model
    )
    return {
        "workflow_id": payload.workflow_id,
        "compare_job_id": compare_job_id,
        "status": "accepted",
        "message": "Compare started",
    }


@app.get("/api/compare/{compare_job_id}/progress")
def compare_progress(compare_job_id: str) -> dict[str, Any]:
    return get_job_progress(compare_job_id)


@app.get("/api/compare/{compare_job_id}/result")
def compare_result(compare_job_id: str) -> dict[str, Any]:
    payload = compare_results.get(compare_job_id)
    if not payload:
        with state_lock:
            job = jobs.get(compare_job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Compare job not found")
            if job.status != "completed":
                raise HTTPException(status_code=409, detail="Compare not finished")
        raise HTTPException(status_code=404, detail="Compare result not found")
    return payload


@app.get("/api/workflows/{workflow_id}")
def workflow_info(workflow_id: str) -> dict[str, Any]:
    return _get_workflow(workflow_id)


@app.get("/api/documents/{document_id}/pdf")
def get_document_pdf(document_id: str, workflow_id: str = Query("default")):
    workflow = _get_workflow(workflow_id)
    doc = next(
        (
            candidate
            for candidate in (workflow.get("file_1"), workflow.get("file_2"))
            if candidate and candidate.get("document_id") == document_id
        ),
        None,
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found in workflow")

    pdf_path = doc.get("normalized_pdf_path") or doc.get("preview_pdf_path")
    if not pdf_path:
        raise HTTPException(status_code=404, detail="Document has no normalized PDF")

    path = Path(pdf_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="PDF not found on server")

    return FileResponse(
        path=str(path), media_type="application/pdf", filename=f"{document_id}.pdf"
    )


def _build_rag_context(compare_payload: dict[str, Any] | None) -> str:
    """
    Xây dựng context gọn, gỡ bỏ trùng lặp dựa trên article + summary.
    """
    if not compare_payload:
        return "Không có dữ liệu so sánh."

    changes = compare_payload.get("changes") or []
    if not changes:
        return "Tài liệu không có thay đổi nào được phát hiện."

    # Dedup: gom nhóm các thay đổi trùng article + summary
    seen: dict[str, dict] = {}
    for ch in changes:
        article = ch.get("article") or "N/A"
        status  = ch.get("status") or ""
        summary = (ch.get("summary") or "").strip()
        key = f"{article}|{status}|{summary[:60]}"
        if key not in seen:
            seen[key] = {
                "article": article, "status": status, "summary": summary, "count": 1,
                "citation_a": (ch.get("evidence") or {}).get("citation_a") or ch.get("citation_a") or "",
                "citation_b": (ch.get("evidence") or {}).get("citation_b") or ch.get("citation_b") or "",
            }
        else:
            seen[key]["count"] += 1

    lines = [f"Tài liệu có {len(changes)} thay đổi ({len(seen)} loại khác nhau):"]
    for item in list(seen.values())[:10]:
        count_note = f" (x{item['count']})" if item["count"] > 1 else ""
        line = f"- {item['article']} [{item['status']}]{count_note}: {item['summary']}"
        if item["citation_a"]:
            line += f"\n  Bản cũ: {item['citation_a'][:100]}"
        if item["citation_b"]:
            line += f"\n  Bản mới: {item['citation_b'][:100]}"
        lines.append(line)

    return "\n".join(lines)


def _build_chat_prompts(message: str, compare_payload: dict[str, Any] | None) -> tuple[str, str]:
    """
    Trả về (system_prompt, user_prompt) tách biệt để gọi /api/chat.
    """
    context_text = _build_rag_context(compare_payload)

    system_prompt = (
        "Bạn là Legal AI Assistant, thông minh và hữu ích.\n"
        "QUY TẮC:\n"
        "1. TRẢ LỜI NGẮN GỌN BẰNG TIẾNG VIỆT 100%.\n"
        "2. Khi người dùng hỏi có bao nhiêu điểm sửa đổi hoặc yêu cầu liệt kê, hãy đếm/tổng hợp dữ liệu so sánh (tính cả số lượng nhân bản xN) và liệt kê rõ ràng.\n"
        "3. Dữ liệu so sánh đưa ra là nền tảng, nếu không có dữ liệu, hãy nói không xác định được."
    )

    # Trả về User Prompt chứa context
    user_context = f"DỮ LIỆU SO SÁNH HIỆN TẠI:\n{context_text}"
    return system_prompt, user_context



@app.post("/api/chat")
def chat_with_document(payload: ChatRequest) -> StreamingResponse:
    """
    Nhận câu hỏi, build RAG prompt từ kết quả so sánh, stream trả lời từ Ollama qua SSE.
    """
    compare_payload: dict[str, Any] | None = None
    if payload.compare_job_id:
        compare_payload = compare_results.get(payload.compare_job_id)

    if not compare_payload:
        # Thử tìm compare result mới nhất của workflow
        with state_lock:
            workflow = workflow_state.get(payload.workflow_id) or {}
            last_job_id = workflow.get("last_compare_job_id")
        if last_job_id:
            compare_payload = compare_results.get(last_job_id)

    system_prompt, user_context = _build_chat_prompts(payload.message, compare_payload)
    model = payload.model

    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "system", "content": user_context})

    for h in payload.history[-8:]: # Giữ tối đa 8 tin nhắn gần nhất để tránh tràn context
        messages.append({"role": h.role, "content": h.content})

    messages.append({"role": "user", "content": payload.message})

    def event_stream():
        try:
            for token in stream_chat_ollama(messages, model=model):
                safe_token = token.replace("\n", "\\n")
                yield f"data: {safe_token}\n\n"
        except ConnectionError as exc:
            yield f"data: [LỖI] {exc}\n\n"
        except Exception as exc:
            yield f"data: [LỖI] Lỗi không xác định: {exc}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
