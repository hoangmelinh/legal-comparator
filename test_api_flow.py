import time
import unittest
from unittest.mock import patch
from pathlib import Path

from fastapi.testclient import TestClient

import app as app_module


def _fake_ingest_document(file_path, doc_id, db_path, cache_root, cache_key=None, progress_callback=None):
    if progress_callback:
        progress_callback(20, "prepare")
        progress_callback(80, "ingest")
        progress_callback(100, "done")
    return {
        "source_hash": f"hash_{doc_id}",
        "normalized_pdf_path": str(file_path),
        "preview_pdf_path": str(file_path),
        "conversion_method": "mock",
        "ingest_status": "completed",
        "warning": "",
    }


def _fake_prepare_document(file_path, cache_root, cache_key=None, progress_callback=None):
    if progress_callback:
        progress_callback(5, "check")
        progress_callback(20, "normalize")
        progress_callback(45, "extract")
        progress_callback(70, "finalize")
    return {
        "source_hash": f"hash_{Path(file_path).stem}",
        "normalized_pdf_path": str(file_path),
        "preview_pdf_path": str(file_path),
        "conversion_method": "mock",
        "text": "mock text",
        "records": [],
        "page_count": 1,
        "total_chars": 9,
        "scan_based": False,
        "weak_extraction": False,
        "can_ingest": True,
        "warning": "",
        "prepared_at": "2026-01-01T00:00:00Z",
        "provenance_schema_version": 1,
    }


def _fake_run_comparison(doc_id_a, doc_id_b, db_path, model, verbose, progress_callback=None):
    if progress_callback:
        progress_callback(25, "scan")
        progress_callback(75, "llm")
        progress_callback(100, "done")
    return [
        {
            "article": "Dieu 1",
            "status": "MODIFIED",
            "change_type": "Sua noi dung",
            "summary": "Noi dung thay doi.",
            "citation_a": "Ban cu",
            "citation_b": "Ban moi",
            "text_a": "Ban cu",
            "text_b": "Ban moi",
            "citation_anchor_a": {"page_start": 1, "anchors": [{"page": 1, "bbox": [10, 10, 40, 20]}]},
            "citation_anchor_b": {"page_start": 1, "anchors": [{"page": 1, "bbox": [12, 12, 42, 24]}]},
            "pdf_anchor_a": {"page_start": 1, "source_hash": "hash_a", "anchors": [{"page": 1, "bbox": [0, 0, 100, 20]}]},
            "pdf_anchor_b": {"page_start": 1, "source_hash": "hash_b", "anchors": [{"page": 1, "bbox": [0, 0, 100, 20]}]},
            "highlight_anchors_a": [{"page": 1, "bbox": [10, 10, 40, 20]}],
            "highlight_anchors_b": [{"page": 1, "bbox": [12, 12, 42, 24]}],
            "changed_spans_a": [{"text": "cu", "start": 4, "end": 6}],
            "changed_spans_b": [{"text": "moi", "start": 4, "end": 7}],
        }
        ]


class ApiFlowTests(unittest.TestCase):
    def setUp(self):
        app_module.jobs.clear()
        app_module.workflow_state.clear()
        app_module.compare_results.clear()
        self.client = TestClient(app_module.app)

    def tearDown(self):
        app_module.jobs.clear()
        app_module.workflow_state.clear()
        app_module.compare_results.clear()

    def _wait_for_job(self, job_id, timeout=3.0):
        started = time.time()
        while time.time() - started < timeout:
            response = self.client.get(f"/api/progress/{job_id}")
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            if payload["status"] in {"completed", "failed"}:
                return payload
            time.sleep(0.05)
        self.fail(f"Job {job_id} did not complete in time")

    def test_upload_requires_file_1_first(self):
        response = self.client.post(
            "/api/documents/upload?slot=file_2&workflow_id=wf_seq",
            files={"file": ("b.txt", b"hello", "text/plain")},
        )
        self.assertEqual(response.status_code, 409)

    def test_empty_upload_is_rejected(self):
        response = self.client.post(
            "/api/documents/upload?slot=file_1&workflow_id=wf_empty",
            files={"file": ("a.txt", b"", "text/plain")},
        )
        self.assertEqual(response.status_code, 400)

    def test_compare_payload_and_workflow_reset(self):
        workflow_id = "wf_full"

        with patch.object(app_module, "prepare_document", side_effect=_fake_prepare_document), patch.object(
            app_module, "ingest_document", side_effect=_fake_ingest_document
        ), patch.object(
            app_module, "run_comparison", side_effect=_fake_run_comparison
        ):
            upload_a = self.client.post(
                f"/api/documents/upload?slot=file_1&workflow_id={workflow_id}",
                files={"file": ("a.txt", b"old", "text/plain")},
            )
            self.assertEqual(upload_a.status_code, 200)
            job_a = upload_a.json()["job_id"]
            self.assertEqual(self._wait_for_job(job_a)["status"], "completed")

            upload_b = self.client.post(
                f"/api/documents/upload?slot=file_2&workflow_id={workflow_id}",
                files={"file": ("b.txt", b"new", "text/plain")},
            )
            self.assertEqual(upload_b.status_code, 200)
            job_b = upload_b.json()["job_id"]
            self.assertEqual(self._wait_for_job(job_b)["status"], "completed")

            compare = self.client.post("/api/compare", json={"workflow_id": workflow_id})
            self.assertEqual(compare.status_code, 200)
            compare_job_id = compare.json()["compare_job_id"]
            self.assertEqual(self._wait_for_job(compare_job_id)["status"], "completed")

            result = self.client.get(f"/api/compare/{compare_job_id}/result")
            self.assertEqual(result.status_code, 200)
            payload = result.json()

            self.assertIn(f"workflow_id={workflow_id}", payload["document_a"]["pdf_url"])
            self.assertIn(f"workflow_id={workflow_id}", payload["document_b"]["pdf_url"])
            self.assertEqual(len(payload["changes"][0]["highlight_anchors_a"]), 1)
            self.assertEqual(payload["report"]["modified"], 1)
            self.assertEqual(payload["changes"][0]["evidence"]["citation_a"], "Ban cu")
            self.assertEqual(payload["changes"][0]["evidence"]["citation_b"], "Ban moi")

            replace_a = self.client.post(
                f"/api/documents/upload?slot=file_1&workflow_id={workflow_id}",
                files={"file": ("a2.txt", b"changed", "text/plain")},
            )
            self.assertEqual(replace_a.status_code, 200)
            replace_job = replace_a.json()["job_id"]
            self.assertEqual(self._wait_for_job(replace_job)["status"], "completed")

            workflow = self.client.get(f"/api/workflows/{workflow_id}")
            self.assertEqual(workflow.status_code, 200)
            workflow_payload = workflow.json()
            self.assertIsNone(workflow_payload["file_2"])
            self.assertIsNone(workflow_payload["last_compare_job_id"])
            self.assertIsNotNone(workflow_payload["file_1"])


if __name__ == "__main__":
    unittest.main()
