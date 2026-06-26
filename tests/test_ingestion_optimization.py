import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import app as app_module
from src.core import ingestion


class FakeLegalVectorDB:
    last_call = None

    def __init__(self, db_name="legal_data.json"):
        self.db_name = db_name
        self.last_insert_timing = {}

    def insert_legal_chunks(self, chunks, metadata_list, doc_id_to_clear=None):
        FakeLegalVectorDB.last_call = {
            "chunks": chunks,
            "metadata_list": metadata_list,
            "doc_id_to_clear": doc_id_to_clear,
        }
        self.last_insert_timing = {
            "embedding_time_sec": 0.012,
            "upsert_time_sec": 0.003,
        }
        return [
            {
                "__id__": f"{metadata_list[0]['doc_id']}_0" if metadata_list else "empty",
                "__vector__": [0.1],
                "content": chunks[0] if chunks else {},
                "metadata": metadata_list[0] if metadata_list else {},
            }
        ]


def _analysis_payload(text: str):
    return {
        "text": text,
        "records": [
            {
                "content": text,
                "canonical": text,
                "retrieval": text.lower(),
                "char_start": 0,
                "char_end": len(text),
                "page_number": 1,
                "source_file": "dummy.pdf",
                "bbox": [10, 10, 20, 20],
                "text_items": [{"text": text, "bbox": [10, 10, 20, 20]}],
                "block_index": 0,
                "line_index": 0,
            }
        ],
        "page_count": 1,
        "total_chars": len(text),
        "average_chars_per_page": float(len(text)),
        "scan_based": False,
        "weak_extraction": False,
        "can_ingest": True,
        "warning": None,
    }


def _chunker(text, doc_id="doc1", records=None):
    return [
        {
            "doc_id": doc_id,
            "article": "Dieu 1",
            "content": text,
            "page_start": 1,
            "page_end": 1,
            "source_anchors": records or [],
        }
    ]


class IngestionOptimizationTests(unittest.TestCase):
    def test_prepare_cache_hit_skips_convert_and_extract(self):
        with tempfile.TemporaryDirectory() as td:
            cache_root = Path(td) / "cache"
            docx_path = Path(td) / "sample.docx"
            docx_path.write_bytes(b"dummy-docx")

            def _mock_convert(source_path, destination_pdf):
                Path(destination_pdf).write_bytes(b"%PDF-1.4 mock")
                return "mock_docx"

            with (
                patch("src.core.ingestion.convert_to_pdf", side_effect=_mock_convert) as mock_convert,
                patch(
                    "src.core.ingestion._extract_text_and_analyse_pdf",
                    side_effect=lambda _: _analysis_payload("DOCX extracted text"),
                ) as mock_extract,
            ):
                first = ingestion.prepare_document(
                    file_path=str(docx_path),
                    cache_root=cache_root,
                    cache_key="same_doc",
                )
                second = ingestion.prepare_document(
                    file_path=str(docx_path),
                    cache_root=cache_root,
                    cache_key="same_doc",
                )

            self.assertEqual(mock_convert.call_count, 1)
            self.assertEqual(mock_extract.call_count, 1)
            self.assertFalse(first["timing"]["cache_hit"])
            self.assertTrue(second["timing"]["cache_hit"])
            self.assertEqual(second["text"], "DOCX extracted text")

    def test_ingest_pdf_docx_txt_preserve_text_and_metadata(self):
        cases = [
            (".pdf", "PDF content", None),
            (".docx", "DOCX content", "mock_docx"),
            (".txt", "TXT content", "mock_txt"),
        ]

        for ext, text_value, conversion_method in cases:
            with tempfile.TemporaryDirectory() as td:
                cache_root = Path(td) / "cache"
                db_path = Path(td) / "db.json"
                file_path = Path(td) / f"sample{ext}"
                file_path.write_bytes(b"dummy")

                def _mock_convert(source_path, destination_pdf):
                    Path(destination_pdf).write_bytes(b"%PDF-1.4 mock")
                    return conversion_method or "unused"

                patchers = [
                    patch("src.core.ingestion.LegalVectorDB", FakeLegalVectorDB),
                    patch("src.core.ingestion.structural_chunking", side_effect=_chunker),
                    patch(
                        "src.core.ingestion._extract_text_and_analyse_pdf",
                        side_effect=lambda _: _analysis_payload(text_value),
                    ),
                ]

                with patchers[0], patchers[1], patchers[2]:
                    if ext != ".pdf":
                        with patch("src.core.ingestion.convert_to_pdf", side_effect=_mock_convert):
                            result = ingestion.ingest_document(
                                file_path=str(file_path),
                                doc_id=f"doc_{ext[1:]}",
                                db_path=str(db_path),
                                cache_root=cache_root,
                            )
                    else:
                        result = ingestion.ingest_document(
                            file_path=str(file_path),
                            doc_id="doc_pdf",
                            db_path=str(db_path),
                            cache_root=cache_root,
                        )

                self.assertEqual(result["text"], text_value)
                self.assertEqual(result["chunks_count"], 1)
                self.assertIn(result["ingest_status"], {"completed", "completed_with_warning"})
                self.assertTrue(result.get("source_hash"))
                self.assertTrue(result.get("normalized_pdf_path"))
                self.assertTrue(result.get("preview_pdf_path"))
                self.assertIn("timing", result)
                self.assertIn("chunk_time_sec", result["timing"])
                self.assertIn("embed_time_sec", result["timing"])

    def test_chunk_count_stable_when_reusing_prepared_doc(self):
        prepared_doc = {
            "source_path": "dummy",
            "source_hash": "hash_1",
            "cache_key": "cache_1",
            "original_extension": ".txt",
            "normalized_pdf_path": "dummy.pdf",
            "preview_pdf_path": "dummy.pdf",
            "conversion_method": "mock",
            "text_cache_path": "dummy.txt",
            "records_cache_path": "dummy.json",
            "records_meta_path": "dummy.meta.json",
            "text": "line 1\nline 2\nline 3",
            "records": [{"content": "line 1"}, {"content": "line 2"}, {"content": "line 3"}],
            "page_count": 1,
            "total_chars": 18,
            "scan_based": False,
            "weak_extraction": False,
            "can_ingest": True,
            "warning": None,
            "prepared_at": "2026-01-01T00:00:00Z",
            "provenance_schema_version": 1,
            "timing": {"normalize_time_sec": 0.1, "parse_time_sec": 0.2, "cache_hit": False},
        }

        def _three_chunks(text, doc_id="doc1", records=None):
            return [
                {"doc_id": doc_id, "article": "A", "content": "a", "page_start": 1, "page_end": 1},
                {"doc_id": doc_id, "article": "B", "content": "b", "page_start": 1, "page_end": 1},
                {"doc_id": doc_id, "article": "C", "content": "c", "page_start": 1, "page_end": 1},
            ]

        with tempfile.TemporaryDirectory() as td:
            cache_root = Path(td) / "cache"
            db_path = Path(td) / "db.json"
            with (
                patch("src.core.ingestion.LegalVectorDB", FakeLegalVectorDB),
                patch("src.core.ingestion.structural_chunking", side_effect=_three_chunks),
            ):
                r1 = ingestion.ingest_document(
                    file_path="dummy.txt",
                    doc_id="doc_x",
                    db_path=str(db_path),
                    cache_root=cache_root,
                    prepared_doc=prepared_doc,
                )
                r2 = ingestion.ingest_document(
                    file_path="dummy.txt",
                    doc_id="doc_x",
                    db_path=str(db_path),
                    cache_root=cache_root,
                    prepared_doc=prepared_doc,
                )

        self.assertEqual(r1["chunks_count"], 3)
        self.assertEqual(r2["chunks_count"], 3)

    def test_ingest_skips_prepare_when_prepared_doc_provided_and_is_faster(self):
        prepared_doc = {
            "source_path": "dummy",
            "source_hash": "hash_2",
            "cache_key": "cache_2",
            "original_extension": ".txt",
            "normalized_pdf_path": "dummy.pdf",
            "preview_pdf_path": "dummy.pdf",
            "conversion_method": "mock",
            "text_cache_path": "dummy.txt",
            "records_cache_path": "dummy.json",
            "records_meta_path": "dummy.meta.json",
            "text": "abc",
            "records": [{"content": "abc"}],
            "page_count": 1,
            "total_chars": 3,
            "scan_based": False,
            "weak_extraction": False,
            "can_ingest": True,
            "warning": None,
            "prepared_at": "2026-01-01T00:00:00Z",
            "provenance_schema_version": 1,
            "timing": {"normalize_time_sec": 0.1, "parse_time_sec": 0.1, "cache_hit": False},
        }

        def _slow_prepare(*args, **kwargs):
            time.sleep(0.2)
            return prepared_doc

        with tempfile.TemporaryDirectory() as td:
            cache_root = Path(td) / "cache"
            db_path = Path(td) / "db.json"
            with (
                patch("src.core.ingestion.LegalVectorDB", FakeLegalVectorDB),
                patch("src.core.ingestion.structural_chunking", side_effect=_chunker),
                patch("src.core.ingestion.prepare_document", side_effect=_slow_prepare),
            ):
                t0 = time.perf_counter()
                ingestion.ingest_document(
                    file_path="dummy.txt",
                    doc_id="doc_slow",
                    db_path=str(db_path),
                    cache_root=cache_root,
                )
                slow_time = time.perf_counter() - t0

                t1 = time.perf_counter()
                ingestion.ingest_document(
                    file_path="dummy.txt",
                    doc_id="doc_fast",
                    db_path=str(db_path),
                    cache_root=cache_root,
                    prepared_doc=prepared_doc,
                )
                fast_time = time.perf_counter() - t1

        self.assertLess(fast_time, slow_time)


class AppProcessFlowTests(unittest.TestCase):
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
        start = time.time()
        while time.time() - start < timeout:
            response = self.client.get(f"/api/progress/{job_id}")
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            if payload["status"] in {"completed", "failed"}:
                return payload
            time.sleep(0.05)
        self.fail("Job timeout")

    def test_upload_pipeline_no_longer_calls_prepare_twice(self):
        def _fake_ingest(
            file_path,
            doc_id,
            db_path,
            cache_root,
            cache_key=None,
            prepared_doc=None,
            progress_callback=None,
        ):
            if progress_callback:
                progress_callback(20, "prepare")
                progress_callback(90, "ingest")
                progress_callback(100, "done")
            return {
                "source_hash": f"hash_{doc_id}",
                "cache_key": cache_key or doc_id,
                "original_extension": ".txt",
                "normalized_pdf_path": str(file_path),
                "preview_pdf_path": str(file_path),
                "conversion_method": "mock",
                "ingest_status": "completed",
                "warning": "",
                "text_cache_path": str(file_path),
                "records_cache_path": str(file_path),
                "records_meta_path": str(file_path),
                "vector_cache_path": str(file_path),
                "chunk_cache_path": str(file_path),
                "timing": {
                    "normalize_time_sec": 0.1,
                    "parse_time_sec": 0.1,
                    "chunk_time_sec": 0.1,
                    "embed_time_sec": 0.1,
                    "vector_insert_time_sec": 0.1,
                    "total_ingest_time_sec": 0.5,
                },
            }

        with (
            patch.object(
                app_module,
                "prepare_document",
                side_effect=RuntimeError("prepare should not be called"),
                create=True,
            ),
            patch.object(app_module, "ingest_document", side_effect=_fake_ingest),
        ):
            upload = self.client.post(
                "/api/documents/upload?slot=file_1&workflow_id=wf_no_prepare",
                files={"file": ("a.txt", b"hello", "text/plain")},
            )
            self.assertEqual(upload.status_code, 200)
            job = self._wait_for_job(upload.json()["job_id"])
            self.assertEqual(job["status"], "completed")


if __name__ == "__main__":
    unittest.main()
