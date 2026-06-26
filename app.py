from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import tempfile
import threading
import uuid
import time
import unicodedata
from collections import Counter
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
from src.core.ingestion import ingest_document
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
chat_response_cache: dict[str, str] = {}
chat_artifact_cache: dict[str, dict[str, Any]] = {}
chat_retrieval_cache: dict[str, list[dict[str, Any]]] = {}
chat_trace_cache: dict[str, dict[str, Any]] = {}

NO_EVIDENCE_ANSWER = "Chưa đủ dữ liệu trong kết quả so sánh để kết luận."
CHAT_SENSITIVE_TERMS = (
    "thanh toán",
    "thanh toan",
    "phạt",
    "phat",
    "bồi thường",
    "boi thuong",
    "bảo mật",
    "bao mat",
    "chấm dứt",
    "cham dut",
    "gia hạn",
    "gia han",
    "thời hạn",
    "thoi han",
    "nghĩa vụ",
    "nghia vu",
    "quyền",
    "quyen",
    "số tiền",
    "so tien",
    "phần trăm",
    "phan tram",
    "tỷ lệ",
    "ty le",
    "bên a",
    "ben a",
    "bên b",
    "ben b",
    "đơn phương",
    "don phuong",
)
LEGAL_SYNONYMS = {
    "thanh toán": ["trả tiền", "chi trả", "tạm ứng", "quyết toán", "công nợ"],
    "thanh toan": ["tra tien", "chi tra", "tam ung", "quyet toan", "cong no"],
    "chấm dứt": ["kết thúc", "hủy bỏ", "đơn phương", "ngừng hợp đồng"],
    "cham dut": ["ket thuc", "huy bo", "don phuong", "ngung hop dong"],
    "phạt": ["phạt vi phạm", "chế tài", "bồi thường", "mức phạt"],
    "phat": ["phat vi pham", "che tai", "boi thuong", "muc phat"],
    "bảo mật": ["bí mật", "thông tin mật", "không tiết lộ"],
    "bao mat": ["bi mat", "thong tin mat", "khong tiet lo"],
    "nghĩa vụ": ["trách nhiệm", "phải thực hiện", "cam kết"],
    "nghia vu": ["trach nhiem", "phai thuc hien", "cam ket"],
    "thời hạn": ["thời gian", "số ngày", "số tháng", "kỳ hạn"],
    "thoi han": ["thoi gian", "so ngay", "so thang", "ky han"],
    "số tiền": ["giá trị", "chi phí", "đơn giá", "phí", "thù lao"],
    "so tien": ["gia tri", "chi phi", "don gia", "phi", "thu lao"],
    "tỷ lệ": ["phần trăm", "%", "mức"],
    "ty le": ["phan tram", "%", "muc"],
}
CHAT_STOPWORDS = {
    "co",
    "có",
    "bao",
    "nhieu",
    "nhiêu",
    "dieu",
    "điều",
    "khoan",
    "khoản",
    "nao",
    "nào",
    "hay",
    "la",
    "là",
    "gi",
    "gì",
    "ve",
    "về",
    "cac",
    "các",
    "nhung",
    "những",
    "bi",
    "bị",
    "duoc",
    "được",
    "thay",
    "doi",
    "đổi",
    "thế",
    "the",
    "nào",
    "nao",
}


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
        "records_meta_path",
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

            result = ingest_document(
                file_path=str(source_path),
                doc_id=document_id,
                db_path=DB_PATH,
                cache_root=CACHE_ROOT,
                cache_key=document_id,
                progress_callback=progress_cb,
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
                    "text_cache_path": result.get("text_cache_path"),
                    "records_cache_path": result.get("records_cache_path"),
                    "records_meta_path": result.get("records_meta_path"),
                    "vector_cache_path": result.get("vector_cache_path"),
                    "chunk_cache_path": result.get("chunk_cache_path"),
                    "convert_status": result.get("conversion_method"),
                    "ingest_status": result.get("ingest_status"),
                    "warning_message": result.get("warning"),
                    "metadata": result,
                    "prepared_metadata": result,
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
            try:
                _get_chat_artifacts(job_id, payload)
            except Exception:
                pass

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

def _chat_env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _chat_env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _chat_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _perf_chat(message: str) -> None:
    print(f"[PERF][chat] {message}")


def _fold_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value or "")
    without_marks = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    return re.sub(r"\s+", " ", without_marks.lower()).strip()


def _question_tokens(value: str) -> set[str]:
    folded = _fold_text(value)
    return {
        token
        for token in re.findall(r"[\w%]+", folded, flags=re.UNICODE)
        if len(token) >= 2 and token not in CHAT_STOPWORDS
    }


def _short_text(value: str, max_chars: int = 700) -> str:
    value = re.sub(r"\s+", " ", value or "").strip()
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 3)].rstrip() + "..."


def _stable_change_id(index: int) -> str:
    return f"CHG-{index:03d}"


def _result_fingerprint(compare_payload: dict[str, Any] | None) -> str:
    changes = (compare_payload or {}).get("changes") or []
    raw = json.dumps(
        [
            {
                "id": change.get("id"),
                "status": change.get("status"),
                "article": change.get("article"),
                "summary": change.get("summary"),
            }
            for change in changes
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _chat_cache_key(
    compare_job_id: str | None, question: str, compare_payload: dict[str, Any] | None
) -> str:
    job_key = compare_job_id or (compare_payload or {}).get("compare_job_id") or "latest"
    raw = f"{job_key}|{_result_fingerprint(compare_payload)}|{_fold_text(question)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _artifact_cache_key(
    compare_job_id: str | None, compare_payload: dict[str, Any] | None
) -> str:
    job_key = compare_job_id or (compare_payload or {}).get("compare_job_id") or "latest"
    return f"{job_key}|{_result_fingerprint(compare_payload)}"


def _clean_chat_answer(answer: str, max_chars: int = 2400) -> str:
    cleaned = _strip_technical_citations(answer or "")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 3].rstrip() + "..."
    return cleaned or NO_EVIDENCE_ANSWER


def _strip_technical_citations(answer: str) -> str:
    cleaned = answer or ""
    cleaned = re.sub(r"\[[\s,;]*(?:CHG-\d{1,6}[\s,;]*)+\]", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\bCHANGE_ID\s*:\s*CHG-\d{1,6}\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\bCHG-\d{1,6}\b", "", cleaned, flags=re.I)
    return cleaned


def _chat_llm_options(question_type: str) -> dict[str, Any]:
    is_fast_answer = question_type in {
        "count_changes",
        "list_changes",
        "summary_overview",
        "filter_by_status",
        "filter_by_topic",
        "filter_by_article",
        "risk_overview_light",
    }
    return {
        "temperature": _chat_env_float("CHAT_TEMPERATURE", 0.1),
        "top_p": _chat_env_float("CHAT_TOP_P", 0.8),
        "num_ctx": _chat_env_int(
            "CHAT_FAST_NUM_CTX" if is_fast_answer else "CHAT_NUM_CTX",
            1024 if is_fast_answer else 2048,
            minimum=512,
        ),
        "num_predict": _chat_env_int(
            "CHAT_FAST_NUM_PREDICT" if is_fast_answer else "CHAT_NUM_PREDICT",
            160 if is_fast_answer else 256,
            minimum=64,
        ),
    }


def _question_topics(question: str) -> set[str]:
    folded = _fold_text(question)
    topics = set()
    for topic, synonyms in LEGAL_SYNONYMS.items():
        terms = [_fold_text(topic)] + [_fold_text(item) for item in synonyms]
        if any(term and term in folded for term in terms):
            topics.add(_fold_text(topic))
    if "phan tram" in folded or "%" in folded:
        topics.add("ty le")
    return topics


def _topic_display_name(topic: str) -> str:
    return {
        "thanh toan": "thanh toán",
        "cham dut": "chấm dứt",
        "phat": "phạt vi phạm",
        "bao mat": "bảo mật",
        "nghia vu": "nghĩa vụ",
        "thoi han": "thời hạn",
        "so tien": "số tiền",
        "ty le": "tỷ lệ",
    }.get(_fold_text(topic), topic)


def _format_article_refs(cards: list[dict[str, Any]], limit: int = 6) -> str:
    articles: list[str] = []
    seen: set[str] = set()
    for card in cards:
        article = (card.get("article") or "").strip()
        if not article:
            continue
        folded = _fold_text(article)
        if folded in seen:
            continue
        seen.add(folded)
        articles.append(article)
        if len(articles) >= limit:
            break
    return ", ".join(articles)


def _dedupe_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for card in cards:
        key = card.get("change_id") or f"{card.get('article')}|{card.get('summary')}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(card)
    return deduped


def _card_evidence_ref(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "change_id": card.get("change_id"),
        "article": card.get("article"),
        "breadcrumb": card.get("breadcrumb"),
        "page_a": card.get("page_a"),
        "page_b": card.get("page_b"),
        "source_anchor_a": card.get("source_anchor_a"),
        "source_anchor_b": card.get("source_anchor_b"),
        "evidence_a": card.get("evidence_a"),
        "evidence_b": card.get("evidence_b"),
    }


def _store_chat_trace(
    cache_key: str,
    compare_job_id: str | None,
    question: str,
    question_type: str,
    fast_path: bool,
    llm_called: bool,
    cards: list[dict[str, Any]],
    answer: str,
) -> None:
    chat_trace_cache[cache_key] = {
        "compare_job_id": compare_job_id,
        "question": question,
        "question_type": question_type,
        "fast_path": fast_path,
        "llm_called": llm_called,
        "evidence_refs": [_card_evidence_ref(card) for card in cards],
        "answer": answer,
    }


def _card_matches_topic(card: dict[str, Any], topic: str) -> bool:
    searchable = card.get("searchable", "")
    terms = [_fold_text(topic)] + [
        _fold_text(item) for item in LEGAL_SYNONYMS.get(topic, [])
    ]
    return any(term and term in searchable for term in terms)


def _get_chat_artifacts(
    compare_job_id: str | None, compare_payload: dict[str, Any] | None
) -> dict[str, Any]:
    cache_key = _artifact_cache_key(compare_job_id, compare_payload)
    cached = chat_artifact_cache.get(cache_key)
    if cached:
        return cached

    cards = _build_change_cards(compare_payload)
    status_index: dict[str, list[dict[str, Any]]] = {}
    article_index: dict[str, list[dict[str, Any]]] = {}
    topic_index: dict[str, list[dict[str, Any]]] = {}
    legal_signal_index: dict[str, list[dict[str, Any]]] = {}
    important_cards: list[dict[str, Any]] = []

    for card in cards:
        status_index.setdefault(card.get("status") or "", []).append(card)
        article_index.setdefault(_fold_text(card.get("article") or ""), []).append(card)
        for topic in LEGAL_SYNONYMS:
            folded_topic = _fold_text(topic)
            if _card_matches_topic(card, folded_topic):
                topic_index.setdefault(folded_topic, []).append(card)
        if any(
            _card_matches_topic(card, _fold_text(topic)) for topic in CHAT_SENSITIVE_TERMS
        ):
            important_cards.append(card)
        for signal in CHAT_SENSITIVE_TERMS:
            folded_signal = _fold_text(signal)
            if folded_signal and folded_signal in card.get("searchable", ""):
                legal_signal_index.setdefault(folded_signal, []).append(card)

    artifacts = {
        "fingerprint": _result_fingerprint(compare_payload),
        "cards": cards,
        "status_index": status_index,
        "article_index": article_index,
        "topic_index": topic_index,
        "legal_signal_index": legal_signal_index,
        "important_changes": _dedupe_cards(
            sorted(
                important_cards,
                key=lambda card: (
                    0 if card.get("important_change") else 1,
                    0 if card.get("risk_note") else 1,
                    card.get("article") or "",
                ),
            )
        ),
        "summary_candidates": _dedupe_cards(cards[: min(8, len(cards))]),
    }
    chat_artifact_cache[cache_key] = artifacts
    return artifacts


def _status_label(status: str) -> str:
    return {
        "ADDED": "thêm mới",
        "REMOVED": "xóa bỏ",
        "MODIFIED": "sửa đổi",
        "MOVED": "di chuyển",
        "MINOR-MODIFIED": "thay đổi nhỏ",
    }.get((status or "").upper(), status or "không rõ")


def _extract_article_refs(question: str) -> set[str]:
    folded = _fold_text(question)
    return {
        match.group(1)
        for match in re.finditer(r"\b(?:dieu|article|clause)\s*(\d+(?:\.\d+)*)", folded)
    }


def _change_evidence(change: dict[str, Any]) -> tuple[str, str]:
    evidence = change.get("evidence") or {}
    raw = change.get("raw") or {}
    return (
        evidence.get("citation_a")
        or change.get("citation_a")
        or raw.get("citation_a")
        or "",
        evidence.get("citation_b")
        or change.get("citation_b")
        or raw.get("citation_b")
        or "",
    )


def _first_anchor(change: dict[str, Any], key: str) -> dict[str, Any] | None:
    anchors = change.get(key) or []
    return anchors[0] if anchors else None


def _build_change_cards(compare_payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for index, change in enumerate((compare_payload or {}).get("changes") or [], start=1):
        citation_a, citation_b = _change_evidence(change)
        raw = change.get("raw") or {}
        article = change.get("article") or raw.get("article") or "N/A"
        status = (change.get("status") or raw.get("status") or "").upper()
        summary = change.get("summary") or raw.get("summary") or ""
        text_a = change.get("text_a") or raw.get("text_a") or citation_a
        text_b = change.get("text_b") or raw.get("text_b") or citation_b
        searchable = " ".join(
            str(part or "")
            for part in (
                article,
                status,
                change.get("change_type"),
                summary,
                raw.get("exact_difference"),
                raw.get("risk_note"),
                citation_a,
                citation_b,
                text_a,
                text_b,
                raw.get("match_reason"),
            )
        )
        cards.append(
            {
                "change_id": _stable_change_id(index),
                "source_id": change.get("id"),
                "article": article,
                "clause": raw.get("clause") or raw.get("clause_id") or "",
                "breadcrumb": raw.get("breadcrumb") or article,
                "status": status,
                "status_label": _status_label(status),
                "summary": summary,
                "exact_difference": raw.get("exact_difference") or "",
                "risk_note": raw.get("risk_note") or "",
                "important_change": bool(raw.get("important_change")),
                "text_a": text_a or "",
                "text_b": text_b or "",
                "evidence_a": citation_a,
                "evidence_b": citation_b,
                "page_a": change.get("page_a") or raw.get("page_a"),
                "page_b": change.get("page_b") or raw.get("page_b"),
                "source_anchor_a": _first_anchor(change, "highlight_anchors_a"),
                "source_anchor_b": _first_anchor(change, "highlight_anchors_b"),
                "similarity_score": raw.get("similarity_score")
                or change.get("similarity_score"),
                "match_reason": raw.get("match_reason") or change.get("match_reason"),
                "keywords": sorted(_question_tokens(searchable)),
                "metadata": {
                    "change_type": change.get("change_type") or raw.get("change_type"),
                    "highlight_mode": change.get("highlight_mode"),
                    "decision_source": raw.get("decision_source")
                    or change.get("decision_source"),
                    "source_hash_a": raw.get("source_hash_a"),
                    "source_hash_b": raw.get("source_hash_b"),
                },
                "searchable": _fold_text(searchable),
            }
        )
    return cards


def _detect_question_type(question: str) -> tuple[str, str | None]:
    folded = _fold_text(question)
    status = None
    if any(term in folded for term in ("them", "them moi", "added")):
        status = "ADDED"
    elif any(term in folded for term in ("xoa", "removed", "xoa bo")):
        status = "REMOVED"
    elif any(term in folded for term in ("di chuyen", "moved")):
        status = "MOVED"
    elif any(term in folded for term in ("sua", "chinh sua", "modified")):
        status = "MODIFIED"
    elif any(term in folded for term in ("nho", "minor")):
        status = "MINOR-MODIFIED"

    if any(term in folded for term in ("bao nhieu", "tong cong", "so luong", "dem")):
        return "count_changes", status
    if any(
        term in folded
        for term in (
            "liet ke",
            "danh sach",
            "nhung dieu khoan nao",
            "cac dieu khoan nao",
            "co dieu khoan nao",
        )
    ):
        return "list_changes", status
    if any(term in folded for term in ("tom tat", "tong quan", "thay doi chinh")):
        return "summary_overview", status
    if any(term in folded for term in ("rui ro", "can chu y", "dang chu y")):
        return "risk_overview_light", status
    if _extract_article_refs(question):
        return "filter_by_article", status
    if _question_topics(question):
        return "filter_by_topic", status
    if status:
        return "filter_by_status", status
    return "analysis", status


def _filter_cards_by_status(
    cards: list[dict[str, Any]], status: str | None
) -> list[dict[str, Any]]:
    if not status:
        return cards
    return [card for card in cards if card.get("status") == status]


def _format_card_line(card: dict[str, Any]) -> str:
    pages = []
    if card.get("page_a"):
        pages.append(f"trang A {card['page_a']}")
    if card.get("page_b"):
        pages.append(f"trang B {card['page_b']}")
    page_note = f" ({', '.join(pages)})" if pages else ""
    summary = card.get("summary") or card.get("metadata", {}).get("change_type") or ""
    return (
        f"- {card.get('article') or 'N/A'} - "
        f"{card.get('status_label')}{page_note}: {_short_text(summary, 220)}"
    )


def _summarize_card_for_user(card: dict[str, Any], max_chars: int = 220) -> str:
    base = (
        card.get("exact_difference")
        or card.get("summary")
        or card.get("risk_note")
        or card.get("metadata", {}).get("change_type")
        or "Có thay đổi đáng chú ý."
    )
    return _short_text(base, max_chars)


def _follow_up_hint(card: dict[str, Any] | None) -> str:
    if not card:
        return ""
    article = card.get("article") or "điều khoản liên quan"
    return f"Bạn nên kiểm tra kỹ {article} vì đây là phần có thể ảnh hưởng trực tiếp đến cách áp dụng điều khoản."


def _answer_count_changes(
    status_filter: str | None, cards: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    selected = _filter_cards_by_status(cards, status_filter)
    if status_filter:
        if not selected:
            return (
                f"Mình chưa thấy thay đổi nào thuộc nhóm {_status_label(status_filter)} trong kết quả so sánh.",
                [],
            )
        article_refs = _format_article_refs(selected)
        answer = f"Có {len(selected)} điều khoản thuộc nhóm {_status_label(status_filter)}."
        if article_refs:
            answer += f" Các thay đổi này xuất hiện ở {article_refs}."
        return answer, selected[: min(6, len(selected))]

    counts = Counter(card.get("status") for card in cards)
    answer = (
        f"Tổng cộng có {len(cards)} thay đổi đáng chú ý trong kết quả so sánh.\n\n"
        f"- Thêm mới: {counts.get('ADDED', 0)}\n"
        f"- Xóa bỏ: {counts.get('REMOVED', 0)}\n"
        f"- Sửa đổi: {counts.get('MODIFIED', 0)}\n"
        f"- Di chuyển: {counts.get('MOVED', 0)}"
    )
    return answer, cards[: min(8, len(cards))]


def _answer_list_changes(
    status_filter: str | None, cards: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    selected = _filter_cards_by_status(cards, status_filter)
    if not selected:
        label = _status_label(status_filter or "") if status_filter else "phù hợp"
        return f"Mình chưa thấy điều khoản nào thuộc nhóm {label} trong kết quả so sánh.", []

    header = (
        f"Có {len(selected)} điều khoản thuộc nhóm {_status_label(status_filter)}:"
        if status_filter
        else f"Có {len(selected)} thay đổi đáng chú ý trong kết quả so sánh:"
    )
    lines = [header]
    lines.extend(_format_card_line(card) for card in selected[:12])
    return "\n".join(lines), selected[:12]


def _answer_summary_overview(artifacts: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    cards = artifacts.get("cards") or []
    if not cards:
        return NO_EVIDENCE_ANSWER, []

    highlights = artifacts.get("important_changes") or artifacts.get("summary_candidates") or cards
    highlights = highlights[:3]
    counts = Counter(card.get("status") for card in cards)
    lines = [
        f"Kết quả so sánh cho thấy {len(cards)} thay đổi đáng chú ý."
        f" Nổi bật nhất là {counts.get('MODIFIED', 0)} điều khoản sửa đổi,"
        f" {counts.get('ADDED', 0)} điều khoản thêm mới và {counts.get('REMOVED', 0)} điều khoản bị xóa."
    ]
    for card in highlights:
        lines.append(
            f"- {card.get('article') or 'N/A'}: {_summarize_card_for_user(card, 180)}"
        )
    hint = _follow_up_hint(highlights[0] if highlights else None)
    if hint:
        lines.append(f"\n{hint}")
    return "\n".join(lines), highlights


def _answer_risk_overview(artifacts: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    risky_cards = [
        card
        for card in (artifacts.get("important_changes") or [])
        if card.get("important_change") or card.get("risk_note")
    ]
    risky_cards = risky_cards[:4]
    if not risky_cards:
        return (
            "Mình chưa thấy đủ bằng chứng để kết luận có thay đổi rủi ro nổi bật hơn các phần khác.",
            [],
        )

    lines = ["Các thay đổi cần chú ý nhất tập trung ở những điều khoản sau:"]
    for card in risky_cards:
        note = card.get("risk_note") or _summarize_card_for_user(card, 180)
        lines.append(f"- {card.get('article') or 'N/A'}: {_short_text(note, 200)}")
    lines.append(
        "\nBạn nên kiểm tra kỹ các điều khoản này vì chúng có thể ảnh hưởng trực tiếp đến quyền, nghĩa vụ hoặc cách thực hiện hợp đồng."
    )
    return "\n".join(lines), risky_cards


def _answer_topic_question(
    question: str, artifacts: dict[str, Any], status_filter: str | None = None
) -> tuple[str, list[dict[str, Any]]]:
    topics = sorted(_question_topics(question))
    if not topics:
        return NO_EVIDENCE_ANSWER, []

    selected: list[dict[str, Any]] = []
    for topic in topics:
        selected.extend(artifacts.get("topic_index", {}).get(topic, []))
        selected.extend(artifacts.get("legal_signal_index", {}).get(topic, []))

    selected = _dedupe_cards(_filter_cards_by_status(selected, status_filter))
    if not selected:
        topic_names = ", ".join(_topic_display_name(topic) for topic in topics)
        return (
            f"Mình chưa thấy thay đổi nào có đủ bằng chứng liên quan đến {topic_names} trong kết quả so sánh.",
            [],
        )

    focus = _topic_display_name(topics[0])
    lines = [f"Có. Phần {focus} có {len(selected)} thay đổi đáng chú ý."]
    for card in selected[:4]:
        lines.append(
            f"- {card.get('article') or 'N/A'}: {_summarize_card_for_user(card, 180)}"
        )
    lines.append(f"\n{_follow_up_hint(selected[0])}")
    return "\n".join(lines), selected[:4]


def _answer_article_question(
    question: str, artifacts: dict[str, Any], status_filter: str | None = None
) -> tuple[str, list[dict[str, Any]]]:
    article_refs = _extract_article_refs(question)
    if not article_refs:
        return NO_EVIDENCE_ANSWER, []

    selected: list[dict[str, Any]] = []
    for ref in article_refs:
        for article_key, cards in (artifacts.get("article_index") or {}).items():
            if ref and ref in article_key:
                selected.extend(cards)

    selected = _dedupe_cards(_filter_cards_by_status(selected, status_filter))
    if not selected:
        return "Mình chưa thấy thay đổi nào khớp với điều khoản bạn đang hỏi.", []

    article_name = selected[0].get("article") or "điều khoản này"
    lines = [f"{article_name} có {len(selected)} thay đổi đáng chú ý trong kết quả so sánh."]
    for card in selected[:4]:
        lines.append(f"- {_summarize_card_for_user(card, 190)}")
    lines.append(f"\n{_follow_up_hint(selected[0])}")
    return "\n".join(lines), selected[:4]


def _answer_status_question(
    status_filter: str | None, artifacts: dict[str, Any]
) -> tuple[str, list[dict[str, Any]]]:
    if not status_filter:
        return NO_EVIDENCE_ANSWER, []
    selected = artifacts.get("status_index", {}).get(status_filter, [])
    if not selected:
        return (
            f"Mình chưa thấy điều khoản nào thuộc nhóm {_status_label(status_filter)} trong kết quả so sánh.",
            [],
        )
    article_refs = _format_article_refs(selected)
    answer = f"Có {len(selected)} điều khoản thuộc nhóm {_status_label(status_filter)}."
    if article_refs:
        answer += f" Các thay đổi này nằm ở {article_refs}."
    return answer, selected[:6]


def _answer_fast_path(
    question: str,
    question_type: str,
    status_filter: str | None,
    artifacts: dict[str, Any],
) -> tuple[str | None, list[dict[str, Any]]]:
    cards = artifacts.get("cards") or []
    if not cards:
        return NO_EVIDENCE_ANSWER, []

    if question_type == "count_changes":
        return _answer_count_changes(status_filter, cards)
    if question_type == "list_changes":
        return _answer_list_changes(status_filter, cards)
    if question_type == "summary_overview":
        return _answer_summary_overview(artifacts)
    if question_type == "risk_overview_light":
        return _answer_risk_overview(artifacts)
    if question_type == "filter_by_topic":
        return _answer_topic_question(question, artifacts, status_filter)
    if question_type == "filter_by_article":
        return _answer_article_question(question, artifacts, status_filter)
    if question_type == "filter_by_status":
        return _answer_status_question(status_filter, artifacts)
    return None, []


def retrieve_relevant_changes(
    question: str,
    cards: list[dict[str, Any]],
    top_k: int | None = None,
    artifacts: dict[str, Any] | None = None,
    status_filter: str | None = None,
) -> list[dict[str, Any]]:
    if not cards:
        return []

    top_k = top_k or _chat_env_int("CHAT_TOP_K", 4)
    q_tokens = _question_tokens(question)
    article_refs = _extract_article_refs(question)
    question_topics = _question_topics(question)
    folded_question = _fold_text(question)
    status_priority = {"MODIFIED": 1.0, "ADDED": 0.9, "REMOVED": 0.9, "MOVED": 0.85}
    sensitive_terms = [
        term for term in CHAT_SENSITIVE_TERMS if _fold_text(term) in folded_question
    ]

    candidate_cards = cards
    if artifacts:
        narrowed: list[dict[str, Any]] = []
        if status_filter:
            narrowed.extend(artifacts.get("status_index", {}).get(status_filter, []))
        for ref in article_refs:
            for article_key, article_cards in (artifacts.get("article_index") or {}).items():
                if ref and ref in article_key:
                    narrowed.extend(article_cards)
        for topic in question_topics:
            narrowed.extend(artifacts.get("topic_index", {}).get(topic, []))
            narrowed.extend(artifacts.get("legal_signal_index", {}).get(topic, []))
        if narrowed:
            candidate_cards = _dedupe_cards(narrowed)

    scored = []
    for card in candidate_cards:
        article_folded = _fold_text(card.get("article", ""))
        breadcrumb_folded = _fold_text(card.get("breadcrumb", ""))
        structural_score = 0.0
        if any(ref and ref in article_folded for ref in article_refs):
            structural_score = 1.0
        elif any(ref and ref in breadcrumb_folded for ref in article_refs):
            structural_score = 0.8

        card_tokens = set(card.get("keywords") or [])
        overlap = len(q_tokens & card_tokens)
        keyword_score = overlap / max(1, len(q_tokens)) if q_tokens else 0.0
        semantic_score = keyword_score
        if folded_question and folded_question in card.get("searchable", ""):
            semantic_score = max(semantic_score, 0.85)
        topic_score = 1.0 if any(_card_matches_topic(card, topic) for topic in question_topics) else 0.0
        sensitive_score = (
            1.0
            if any(_fold_text(term) in card.get("searchable", "") for term in sensitive_terms)
            else 0.0
        )
        status_score = status_priority.get(card.get("status"), 0.5)
        final_score = (
            structural_score * 0.35
            + keyword_score * 0.25
            + semantic_score * 0.15
            + topic_score * 0.10
            + sensitive_score * 0.05
            + status_score * 0.05
            + (0.05 if card.get("important_change") else 0.0)
        )
        if final_score > 0.08:
            scored.append((final_score, card))

    scored.sort(key=lambda item: item[0], reverse=True)
    ranked = [card for _, card in scored[:top_k]]
    if ranked:
        return ranked
    if artifacts and artifacts.get("important_changes"):
        return artifacts["important_changes"][:top_k]
    return cards[:top_k]


def _context_for_cards(cards: list[dict[str, Any]], question_type: str) -> str:
    max_chars = _chat_env_int("CHAT_MAX_CONTEXT_CHARS", 4000, minimum=1000)
    summary_only = question_type == "summary_overview" and _chat_env_bool(
        "CHAT_SUMMARY_ONLY_FOR_OVERVIEW", True
    )
    blocks = []
    used_chars = 0
    for card in cards:
        lines = [
            f"CHANGE_ID: {card['change_id']}",
            f"Điều khoản: {card.get('article') or 'N/A'}",
            f"Breadcrumb: {card.get('breadcrumb') or card.get('article') or 'N/A'}",
            f"Trạng thái: {card.get('status')}",
            f"Tóm tắt kỹ thuật: {_short_text(card.get('summary'), 420)}",
        ]
        if card.get("exact_difference"):
            lines.append(
                f"Khác biệt cụ thể: {_short_text(card.get('exact_difference'), 420)}"
            )
        if card.get("risk_note"):
            lines.append(f"Ghi chú tác động: {_short_text(card.get('risk_note'), 220)}")
        if card.get("page_a"):
            lines.append(f"Trang bản A: {card['page_a']}")
        if card.get("page_b"):
            lines.append(f"Trang bản B: {card['page_b']}")
        if card.get("match_reason"):
            lines.append(f"Lý do match: {_short_text(card.get('match_reason'), 220)}")
        if not summary_only:
            if card.get("evidence_a") or card.get("text_a"):
                lines.append(
                    f"Evidence A: {_short_text(card.get('evidence_a') or card.get('text_a'), 520)}"
                )
            if card.get("evidence_b") or card.get("text_b"):
                lines.append(
                    f"Evidence B: {_short_text(card.get('evidence_b') or card.get('text_b'), 520)}"
                )
        block = "\n".join(lines)
        if blocks and used_chars + len(block) > max_chars:
            break
        blocks.append(block)
        used_chars += len(block)
    return "\n\n".join(blocks)


def _build_evidence_chat_prompts(
    message: str, selected_cards: list[dict[str, Any]], question_type: str
) -> tuple[str, str]:
    context_text = _context_for_cards(selected_cards, question_type)
    system_prompt = (
        "Bạn là trợ lý phân tích thay đổi trong văn bản pháp lý.\n"
        "Chỉ được dùng CONTEXT được cung cấp.\n"
        "Không được bịa thêm điều khoản, số liệu, rủi ro hoặc kết luận ngoài context.\n"
        f'Nếu context không đủ, hãy nói đúng câu: "{NO_EVIDENCE_ANSWER}"\n'
        "Trả lời bằng tiếng Việt tự nhiên, ngắn gọn và thực tế.\n"
        "Ưu tiên nêu kết luận trước, sau đó giải thích ngắn gọn.\n"
        "Không hiển thị mã kỹ thuật như CHG-001 hoặc citation dạng [CHG-001].\n"
        "Không dùng giọng quá chắc chắn nếu bằng chứng còn mỏng.\n"
        "Nếu có đủ căn cứ, có thể nêu tác động thực tế đến quyền, nghĩa vụ, thời hạn, thanh toán, phạt, bảo mật hoặc chấm dứt.\n"
        "Giữ câu trả lời trong khoảng 5-8 câu trừ khi người dùng yêu cầu phân tích chi tiết."
    )
    user_context = (
        "CONTEXT KẾT QUẢ SO SÁNH:\n"
        f"{context_text}\n\n"
        f"CÂU HỎI: {message}"
    )
    return system_prompt, user_context


def _sanitize_streaming_chunks(tokens: list[str], tail_chars: int = 24) -> list[str]:
    raw_buffer = ""
    emitted_len = 0
    chunks: list[str] = []
    for token in tokens:
        raw_buffer += token
        stable_text = raw_buffer[:-tail_chars] if len(raw_buffer) > tail_chars else ""
        cleaned_stable = _strip_technical_citations(stable_text)
        if len(cleaned_stable) > emitted_len:
            chunks.append(cleaned_stable[emitted_len:])
            emitted_len = len(cleaned_stable)

    cleaned_full = _strip_technical_citations(raw_buffer)
    if len(cleaned_full) > emitted_len:
        chunks.append(cleaned_full[emitted_len:])
    return [chunk for chunk in chunks if chunk]


@app.post("/api/_chat_legacy_unused")
def chat_with_document(payload: ChatRequest) -> StreamingResponse:
    """
    Nhận câu hỏi, build RAG prompt từ kết quả so sánh, stream trả lời từ Ollama qua SSE.
    """
    started = time.perf_counter()
    load_started = time.perf_counter()
    compare_payload: dict[str, Any] | None = None
    compare_job_id = payload.compare_job_id
    if payload.compare_job_id:
        compare_payload = compare_results.get(payload.compare_job_id)

    if not compare_payload:
        with state_lock:
            workflow = workflow_state.get(payload.workflow_id) or {}
            last_job_id = workflow.get("last_compare_job_id")
        if last_job_id:
            compare_job_id = last_job_id
            compare_payload = compare_results.get(last_job_id)

    load_time = time.perf_counter() - load_started
    cards_started = time.perf_counter()
    cards = _build_change_cards(compare_payload)
    cards_time = time.perf_counter() - cards_started
    question_type, status_filter = _detect_question_type(payload.message)
    cache_enabled = _chat_env_bool("CHAT_CACHE_ENABLED", True)
    cache_key = _chat_cache_key(compare_job_id, payload.message, compare_payload)

    _perf_chat(f"question_type={question_type}")
    _perf_chat(f"compare_result={load_time:.2f}s")
    _perf_chat(f"change_cards={cards_time:.2f}s count={len(cards)}")

    def _sse_escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "")

    def _stream_text(text: str):
        yield f"data: {_sse_escape(text)}\n\n"
        yield "data: [DONE]\n\n"

    if cache_enabled and cache_key in chat_response_cache:
        _perf_chat("cache_hit=true")
        _perf_chat("retrieval=0.00s top_k=0")
        _perf_chat("llm_called=false")
        _perf_chat(f"total={time.perf_counter() - started:.2f}s")
        return StreamingResponse(
            _stream_text(chat_response_cache[cache_key]),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if not cards:
        answer = NO_EVIDENCE_ANSWER
        if cache_enabled:
            chat_response_cache[cache_key] = answer
        _perf_chat("cache_hit=false")
        _perf_chat("retrieval=0.00s top_k=0")
        _perf_chat("llm_called=false")
        _perf_chat(f"total={time.perf_counter() - started:.2f}s")
        return StreamingResponse(
            _stream_text(answer),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if question_type in {"stats", "list"}:
        retrieval_started = time.perf_counter()
        answer = _answer_direct_question(question_type, status_filter, cards)
        if cache_enabled:
            chat_response_cache[cache_key] = answer
        _perf_chat("cache_hit=false")
        _perf_chat(f"retrieval={time.perf_counter() - retrieval_started:.2f}s top_k=0")
        _perf_chat("llm_called=false")
        _perf_chat(f"total={time.perf_counter() - started:.2f}s")
        return StreamingResponse(
            _stream_text(answer),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    retrieval_started = time.perf_counter()
    selected_cards = retrieve_relevant_changes(payload.message, cards)
    retrieval_time = time.perf_counter() - retrieval_started
    _perf_chat("cache_hit=false")
    _perf_chat(f"retrieval={retrieval_time:.2f}s top_k={len(selected_cards)}")

    if not selected_cards:
        answer = NO_EVIDENCE_ANSWER
        if cache_enabled:
            chat_response_cache[cache_key] = answer
        _perf_chat("llm_called=false")
        _perf_chat(f"total={time.perf_counter() - started:.2f}s")
        return StreamingResponse(
            _stream_text(answer),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    system_prompt, user_context = _build_evidence_chat_prompts(
        payload.message, selected_cards, question_type
    )
    model = os.getenv("OLLAMA_MODEL") or payload.model or DEFAULT_MODEL
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": user_context},
        {"role": "user", "content": payload.message},
    ]

    def optimized_event_stream():
        llm_started = time.perf_counter()
        accumulated = ""
        _perf_chat("llm_called=true")
        try:
            for token in stream_chat_ollama(messages, model=model):
                accumulated += token
                yield f"data: {_sse_escape(token)}\n\n"
        except ConnectionError as exc:
            accumulated = f"[LỖI] Không thể kết nối Ollama: {exc}"
            yield f"data: {_sse_escape(accumulated)}\n\n"
        except Exception as exc:
            accumulated = f"[LỖI] Lỗi không xác định khi gọi Chatbot: {exc}"
            yield f"data: {_sse_escape(accumulated)}\n\n"
        finally:
            if accumulated and cache_enabled and not accumulated.startswith("[LỖI]"):
                chat_response_cache[cache_key] = accumulated
            _perf_chat(f"llm_time={time.perf_counter() - llm_started:.2f}s")
            _perf_chat(f"total={time.perf_counter() - started:.2f}s")
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        optimized_event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

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
                # Escape newlines for SSE single-line data format
                safe_token = token.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "")
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


@app.post("/api/chat")
def chat_with_document_fast(payload: ChatRequest) -> StreamingResponse:
    """
    Nhan cau hoi, uu tien tra loi truc tiep tu ket qua so sanh va chi goi Ollama khi can.
    """
    started = time.perf_counter()
    load_started = time.perf_counter()
    compare_payload: dict[str, Any] | None = None
    compare_job_id = payload.compare_job_id
    if payload.compare_job_id:
        compare_payload = compare_results.get(payload.compare_job_id)

    if not compare_payload:
        with state_lock:
            workflow = workflow_state.get(payload.workflow_id) or {}
            last_job_id = workflow.get("last_compare_job_id")
        if last_job_id:
            compare_job_id = last_job_id
            compare_payload = compare_results.get(last_job_id)

    load_time = time.perf_counter() - load_started
    question_type, status_filter = _detect_question_type(payload.message)
    cache_enabled = _chat_env_bool("CHAT_CACHE_ENABLED", True)
    cache_key = _chat_cache_key(compare_job_id, payload.message, compare_payload)

    _perf_chat(f"question_type={question_type}")
    _perf_chat(f"compare_result={load_time:.2f}s")

    def _sse_escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "")

    def _stream_text(text: str):
        yield f"data: {_sse_escape(text)}\n\n"
        yield "data: [DONE]\n\n"

    if cache_enabled and cache_key in chat_response_cache:
        trace = chat_trace_cache.get(cache_key, {})
        _perf_chat("cache_hit=true")
        _perf_chat(f"fast_path={str(trace.get('fast_path', False)).lower()}")
        _perf_chat("change_cards_time=0.00s")
        _perf_chat("retrieval_time=0.00s")
        _perf_chat("top_k=0")
        _perf_chat("context_chars=0")
        _perf_chat(f"llm_called={str(trace.get('llm_called', False)).lower()}")
        _perf_chat(f"total={time.perf_counter() - started:.2f}s")
        return StreamingResponse(
            _stream_text(chat_response_cache[cache_key]),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    artifacts_started = time.perf_counter()
    artifacts = _get_chat_artifacts(compare_job_id, compare_payload)
    cards = artifacts.get("cards") or []
    cards_time = time.perf_counter() - artifacts_started
    _perf_chat(f"change_cards_time={cards_time:.2f}s count={len(cards)}")

    if not cards:
        answer = NO_EVIDENCE_ANSWER
        if cache_enabled:
            chat_response_cache[cache_key] = answer
        _store_chat_trace(
            cache_key,
            compare_job_id,
            payload.message,
            question_type,
            True,
            False,
            [],
            answer,
        )
        _perf_chat("cache_hit=false")
        _perf_chat("fast_path=true")
        _perf_chat("retrieval_time=0.00s")
        _perf_chat("top_k=0")
        _perf_chat("context_chars=0")
        _perf_chat("llm_called=false")
        _perf_chat(f"total={time.perf_counter() - started:.2f}s")
        return StreamingResponse(
            _stream_text(answer),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    retrieval_started = time.perf_counter()
    fast_answer, fast_cards = _answer_fast_path(
        payload.message, question_type, status_filter, artifacts
    )
    fast_retrieval_time = time.perf_counter() - retrieval_started
    if fast_answer is not None:
        answer = _clean_chat_answer(fast_answer)
        if cache_enabled:
            chat_response_cache[cache_key] = answer
        _store_chat_trace(
            cache_key,
            compare_job_id,
            payload.message,
            question_type,
            True,
            False,
            fast_cards,
            answer,
        )
        _perf_chat("cache_hit=false")
        _perf_chat("fast_path=true")
        _perf_chat(f"retrieval_time={fast_retrieval_time:.2f}s")
        _perf_chat(f"top_k={len(fast_cards)}")
        _perf_chat("context_chars=0")
        _perf_chat("llm_called=false")
        _perf_chat(f"total={time.perf_counter() - started:.2f}s")
        return StreamingResponse(
            _stream_text(answer),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    retrieval_started = time.perf_counter()
    retrieval_cache_hit = False
    if cache_enabled and cache_key in chat_retrieval_cache:
        selected_cards = chat_retrieval_cache[cache_key]
        retrieval_cache_hit = True
    else:
        selected_cards = retrieve_relevant_changes(
            payload.message,
            cards,
            artifacts=artifacts,
            status_filter=status_filter,
        )
        if cache_enabled:
            chat_retrieval_cache[cache_key] = selected_cards
    retrieval_time = time.perf_counter() - retrieval_started
    _perf_chat("cache_hit=false")
    _perf_chat("fast_path=false")
    _perf_chat(f"retrieval_time={retrieval_time:.2f}s")
    _perf_chat(f"retrieval_cache_hit={str(retrieval_cache_hit).lower()}")
    _perf_chat(f"top_k={len(selected_cards)}")

    if not selected_cards:
        answer = NO_EVIDENCE_ANSWER
        if cache_enabled:
            chat_response_cache[cache_key] = answer
        _store_chat_trace(
            cache_key,
            compare_job_id,
            payload.message,
            question_type,
            False,
            False,
            [],
            answer,
        )
        _perf_chat("context_chars=0")
        _perf_chat("llm_called=false")
        _perf_chat(f"total={time.perf_counter() - started:.2f}s")
        return StreamingResponse(
            _stream_text(answer),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    system_prompt, user_context = _build_evidence_chat_prompts(
        payload.message, selected_cards, question_type
    )
    context_chars = len(user_context)
    model = os.getenv("OLLAMA_MODEL") or payload.model or DEFAULT_MODEL
    llm_options = _chat_llm_options(question_type)
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "system", "content": user_context})
    for item in payload.history[-4:]:
        if item.role in {"user", "assistant"} and item.content.strip():
            messages.append(
                {"role": item.role, "content": _short_text(item.content, 500)}
            )
    messages.append({"role": "user", "content": payload.message})
    _perf_chat(f"context_chars={context_chars}")

    def optimized_event_stream():
        llm_started = time.perf_counter()
        raw_output = ""
        emitted_len = 0
        _perf_chat("llm_called=true")
        try:
            for token in stream_chat_ollama(messages, model=model, options=llm_options):
                raw_output += token
                stable_text = raw_output[:-24] if len(raw_output) > 24 else ""
                cleaned_stable = _strip_technical_citations(stable_text)
                if len(cleaned_stable) > emitted_len:
                    chunk = cleaned_stable[emitted_len:]
                    emitted_len = len(cleaned_stable)
                    yield f"data: {_sse_escape(chunk)}\n\n"
        except ConnectionError as exc:
            raw_output = f"[ERROR] Khong the ket noi Ollama: {exc}"
            yield f"data: {_sse_escape(raw_output)}\n\n"
        except Exception as exc:
            raw_output = f"[ERROR] Loi khong xac dinh khi goi Chatbot: {exc}"
            yield f"data: {_sse_escape(raw_output)}\n\n"
        finally:
            accumulated = _clean_chat_answer(raw_output)
            if not raw_output.startswith("[ERROR]"):
                cleaned_full = _strip_technical_citations(raw_output)
                if len(cleaned_full) > emitted_len:
                    yield f"data: {_sse_escape(cleaned_full[emitted_len:])}\n\n"
            if accumulated and cache_enabled and not raw_output.startswith("[ERROR]"):
                chat_response_cache[cache_key] = accumulated
            if accumulated and not raw_output.startswith("[ERROR]"):
                _store_chat_trace(
                    cache_key,
                    compare_job_id,
                    payload.message,
                    question_type,
                    False,
                    True,
                    selected_cards,
                    accumulated,
                )
            _perf_chat(f"llm_time={time.perf_counter() - llm_started:.2f}s")
            _perf_chat(f"total={time.perf_counter() - started:.2f}s")
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        optimized_event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

