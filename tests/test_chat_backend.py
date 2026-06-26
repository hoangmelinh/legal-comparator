from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import app as app_module


def _sse_text(response) -> str:
    chunks: list[str] = []
    for line in response.text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            continue
        chunks.append(payload.replace("\\n", "\n").replace("\\\\", "\\"))
    return "".join(chunks)


def _sample_compare_payload() -> dict:
    return {
        "compare_job_id": "job-test",
        "workflow_id": "wf-test",
        "changes": [
            {
                "id": "chg_1",
                "article": "Dieu 5",
                "status": "MODIFIED",
                "summary": "Dieu khoan thanh toan duoc sua theo huong keo dai thoi han.",
                "text_a": "Ben B thanh toan trong 15 ngay.",
                "text_b": "Ben B thanh toan trong 30 ngay.",
                "page_a": 3,
                "page_b": 4,
                "evidence": {
                    "citation_a": "thanh toan trong 15 ngay",
                    "citation_b": "thanh toan trong 30 ngay",
                },
                "highlight_anchors_a": [{"page": 3, "text": "15 ngay"}],
                "highlight_anchors_b": [{"page": 4, "text": "30 ngay"}],
                "raw": {
                    "article": "Dieu 5",
                    "status": "MODIFIED",
                    "summary": "Dieu khoan thanh toan duoc sua theo huong keo dai thoi han.",
                    "exact_difference": "Thoi han thanh toan doi tu 15 ngay sang 30 ngay.",
                    "risk_note": "Lam ben nhan thanh toan cho lau hon.",
                    "important_change": True,
                    "change_type": "Sua noi dung",
                    "breadcrumb": "Dieu 5 - Thanh toan",
                },
            },
            {
                "id": "chg_2",
                "article": "Dieu 7",
                "status": "ADDED",
                "summary": "Bo sung dieu khoan bao mat thong tin.",
                "text_a": "",
                "text_b": "Cac ben phai giu bi mat thong tin giao dich.",
                "page_a": None,
                "page_b": 6,
                "evidence": {
                    "citation_a": "",
                    "citation_b": "giu bi mat thong tin giao dich",
                },
                "highlight_anchors_a": [],
                "highlight_anchors_b": [{"page": 6, "text": "giu bi mat"}],
                "raw": {
                    "article": "Dieu 7",
                    "status": "ADDED",
                    "summary": "Bo sung dieu khoan bao mat thong tin.",
                    "exact_difference": "Them moi nghia vu bao mat thong tin.",
                    "risk_note": "",
                    "important_change": True,
                    "change_type": "Them moi",
                    "breadcrumb": "Dieu 7 - Bao mat",
                },
            },
        ],
        "report_changes": [],
        "report": {"total": 2, "added": 1, "removed": 0, "modified": 1, "moved": 0},
    }


class ChatBackendTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app_module.app)
        app_module.compare_results.clear()
        app_module.workflow_state.clear()
        app_module.chat_response_cache.clear()
        app_module.chat_artifact_cache.clear()
        app_module.chat_retrieval_cache.clear()
        app_module.chat_trace_cache.clear()
        app_module.compare_results["job-test"] = _sample_compare_payload()

    def test_stats_question_uses_fast_path_without_ollama(self):
        with patch.object(app_module, "stream_chat_ollama") as ollama_mock:
            response = self.client.post(
                "/api/chat",
                json={"compare_job_id": "job-test", "message": "Co bao nhieu dieu khoan bi sua?"},
            )

        self.assertEqual(response.status_code, 200)
        answer = _sse_text(response)
        self.assertIn("1", answer)
        self.assertNotIn("CHG-", answer)
        ollama_mock.assert_not_called()

    def test_topic_fast_path_keeps_internal_evidence_refs(self):
        with patch.object(app_module, "stream_chat_ollama") as ollama_mock:
            response = self.client.post(
                "/api/chat",
                json={"compare_job_id": "job-test", "message": "Co thay doi nao ve thanh toan khong?"},
            )

        self.assertEqual(response.status_code, 200)
        answer = _sse_text(response)
        self.assertIn("thanh", answer.lower())
        self.assertNotIn("CHG-", answer)
        ollama_mock.assert_not_called()

        cache_key = app_module._chat_cache_key(  # noqa: SLF001
            "job-test", "Co thay doi nao ve thanh toan khong?", app_module.compare_results["job-test"]
        )
        trace = app_module.chat_trace_cache[cache_key]
        self.assertTrue(trace["evidence_refs"])
        self.assertEqual(trace["evidence_refs"][0]["change_id"], "CHG-001")

    def test_analysis_question_calls_ollama_and_cleans_streamed_answer(self):
        captured = {}

        def fake_stream(messages, model=None, timeout=300, options=None):
            captured["messages"] = messages
            captured["options"] = options
            for token in (
                "Phan thanh toan can chu y [CHG-001]. ",
                "Thoi han thanh toan da doi tu 15 ngay sang 30 ngay.",
            ):
                yield token

        with patch.object(app_module, "stream_chat_ollama", side_effect=fake_stream):
            response = self.client.post(
                "/api/chat",
                json={"compare_job_id": "job-test", "message": "Phan tich thay doi quan trong nhat trong bo ket qua."},
            )

        self.assertEqual(response.status_code, 200)
        answer = _sse_text(response)
        self.assertIn("30 ngay", answer)
        self.assertNotIn("CHG-", answer)
        self.assertNotIn("[CHG-", answer)
        self.assertEqual(captured["options"]["num_predict"], 256)
        prompt_lower = captured["messages"][0]["content"].lower()
        self.assertNotIn("citation change_id", prompt_lower)
        self.assertIn("chg-001", prompt_lower)

    def test_repeated_analysis_question_uses_cache(self):
        call_count = {"value": 0}

        def fake_stream(messages, model=None, timeout=300, options=None):
            call_count["value"] += 1
            yield "Cau tra loi phan tich."

        payload = {
            "compare_job_id": "job-test",
            "message": "Phan tich thay doi quan trong nhat.",
        }
        with patch.object(app_module, "stream_chat_ollama", side_effect=fake_stream):
            first = self.client.post("/api/chat", json=payload)
            second = self.client.post("/api/chat", json=payload)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(call_count["value"], 1)
        self.assertEqual(_sse_text(first), _sse_text(second))


if __name__ == "__main__":
    unittest.main()
