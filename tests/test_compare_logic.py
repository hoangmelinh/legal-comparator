import unittest

import app as app_module
from src.core import comparator


class CompareLogicTests(unittest.TestCase):
    def test_numeric_detection_includes_percent_date_and_units(self):
        self.assertTrue(comparator.has_numeric_diff("Lãi suất 10%", "Lãi suất 12%"))
        self.assertTrue(
            comparator.has_numeric_diff("Hạn đến 01/01/2024", "Hạn đến 02/01/2024")
        )
        self.assertTrue(
            comparator.has_numeric_diff("Khối lượng 10 kg", "Khối lượng 10 g")
        )

    def test_word_level_span_stays_tight_on_numeric_change(self):
        payload = comparator._build_minimal_changed_spans(
            "Phí là 10 đồng", "Phí là 12 đồng"
        )
        spans_a = payload["spans_a"]
        spans_b = payload["spans_b"]

        self.assertEqual(len(spans_a), 1)
        self.assertEqual(len(spans_b), 1)
        self.assertEqual(spans_a[0]["text"], "10")
        self.assertEqual(spans_b[0]["text"], "12")

    def test_prepare_job_payload_filters_minor_changes(self):
        change_list = [
            {
                "article": "Dieu 1",
                "status": "MINOR-MODIFIED",
                "change_type": "Thay đổi nhỏ về câu chữ",
                "summary": "Bỏ qua",
                "citation_a": "",
                "citation_b": "",
                "text_a": "A",
                "text_b": "B",
                "highlight_anchors_a": [{"page": 1, "bbox": [1, 2, 3, 4]}],
                "highlight_anchors_b": [{"page": 1, "bbox": [1, 2, 3, 4]}],
                "changed_spans_a": [
                    {"text": "A", "start": 0, "end": 1, "token_count": 1}
                ],
                "changed_spans_b": [
                    {"text": "B", "start": 0, "end": 1, "token_count": 1}
                ],
                "citation_anchor_a": None,
                "citation_anchor_b": None,
                "pdf_anchor_a": {"doc_id": "doc_a", "page_start": 1},
                "pdf_anchor_b": {"doc_id": "doc_b", "page_start": 1},
            },
            {
                "article": "Dieu 2",
                "status": "MODIFIED",
                "change_type": "Sửa nội dung",
                "summary": "Giữ lại",
                "citation_a": "Cũ",
                "citation_b": "Mới",
                "text_a": "Cũ",
                "text_b": "Mới",
                "highlight_anchors_a": [{"page": 2, "bbox": [10, 20, 30, 40]}],
                "highlight_anchors_b": [{"page": 2, "bbox": [10, 20, 30, 40]}],
                "changed_spans_a": [
                    {"text": "Cũ", "start": 0, "end": 3, "token_count": 1}
                ],
                "changed_spans_b": [
                    {"text": "Mới", "start": 0, "end": 3, "token_count": 1}
                ],
                "citation_anchor_a": None,
                "citation_anchor_b": None,
                "pdf_anchor_a": {"doc_id": "doc_a", "page_start": 2},
                "pdf_anchor_b": {"doc_id": "doc_b", "page_start": 2},
            },
        ]

        payload = app_module._prepare_job_payload(
            change_list=change_list,
            workflow_id="wf_test",
            doc_a={"document_id": "doc_a", "workflow_doc_id": "doc_a"},
            doc_b={"document_id": "doc_b", "workflow_doc_id": "doc_b"},
            compare_job_id="job_1",
        )

        self.assertEqual(len(payload["changes"]), 1)
        self.assertEqual(payload["changes"][0]["id"], "chg_1")
        self.assertEqual(payload["report"]["total"], 1)
        self.assertEqual(payload["report"]["minor_modified"], 0)
        self.assertEqual(payload["changes"][0]["status"], "MODIFIED")

    def test_prepare_job_payload_uses_highlight_page_and_reindexes_ids(self):
        change_list = [
            {
                "article": "Dieu 0",
                "status": "MINOR-MODIFIED",
                "change_type": "Thay đổi nhỏ về câu chữ",
                "summary": "Bỏ qua",
                "highlight_anchors_a": [{"page": 1, "bbox": [0, 0, 1, 1]}],
                "highlight_anchors_b": [{"page": 1, "bbox": [0, 0, 1, 1]}],
            },
            {
                "article": "Dieu 1",
                "status": "MODIFIED",
                "change_type": "Sửa nội dung",
                "summary": "Giữ lại 1",
                "highlight_mode": "inline",
                "highlight_anchors_a": [
                    {
                        "page": 3,
                        "bbox": [10, 20, 30, 40],
                        "kind": "inline",
                        "strike": True,
                    }
                ],
                "highlight_anchors_b": [
                    {
                        "page": 4,
                        "bbox": [11, 21, 31, 41],
                        "kind": "inline",
                        "strike": False,
                    }
                ],
            },
            {
                "article": "Dieu 2",
                "status": "REMOVED",
                "change_type": "Xóa bỏ",
                "summary": "Giữ lại 2",
                "highlight_mode": "block",
                "highlight_anchors_a": [
                    {"page": 7, "bbox": [1, 2, 3, 4], "kind": "block", "strike": True}
                ],
                "highlight_anchors_b": [],
            },
        ]

        payload = app_module._prepare_job_payload(
            change_list=change_list,
            workflow_id="wf_test_page",
            doc_a={"document_id": "doc_a", "workflow_doc_id": "doc_a"},
            doc_b={"document_id": "doc_b", "workflow_doc_id": "doc_b"},
            compare_job_id="job_2",
        )

        self.assertEqual(len(payload["changes"]), 2)
        self.assertEqual(payload["changes"][0]["id"], "chg_1")
        self.assertEqual(payload["changes"][1]["id"], "chg_2")
        self.assertEqual(payload["changes"][0]["page_a"], 3)
        self.assertEqual(payload["changes"][0]["page_b"], 4)
        self.assertEqual(payload["changes"][1]["page_a"], 7)

    def test_clause_highlights_mark_block_and_strike(self):
        provenance = {
            "anchors": [
                {
                    "page": 2,
                    "bbox": [54, 100, 420, 116],
                    "block_index": 2,
                    "line_index": 0,
                },
                {
                    "page": 2,
                    "bbox": [54, 118, 430, 134],
                    "block_index": 2,
                    "line_index": 1,
                },
            ]
        }
        block = comparator._build_clause_highlights(provenance, strike=True)
        self.assertEqual(len(block), 2)
        self.assertEqual(block[0]["kind"], "block")
        self.assertTrue(block[0]["strike"])

    def test_modified_highlight_mode_switches_by_scope(self):
        token_stats_small = {
            "tokens_a": [1] * 12,
            "tokens_b": [1] * 12,
            "changed_a": 1,
            "changed_b": 1,
        }
        payload_small = {
            "changed_spans_a": [{"text": "10"}],
            "changed_spans_b": [{"text": "12"}],
        }
        self.assertFalse(
            comparator._should_use_block_highlight_for_modified(
                True, token_stats_small, payload_small
            )
        )
        self.assertFalse(
            comparator._should_use_block_highlight_for_modified(
                False, token_stats_small, payload_small
            )
        )

        token_stats_big = {
            "tokens_a": [1] * 12,
            "tokens_b": [1] * 12,
            "changed_a": 7,
            "changed_b": 7,
        }
        payload_big = {
            "changed_spans_a": [{"text": "A"}] * 6,
            "changed_spans_b": [{"text": "B"}] * 6,
        }
        self.assertTrue(
            comparator._should_use_block_highlight_for_modified(
                False, token_stats_big, payload_big
            )
        )


if __name__ == "__main__":
    unittest.main()
