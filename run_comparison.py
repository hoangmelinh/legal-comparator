"""
run_comparison.py
─────────────────────────────────────────────────────────
CLI tool để:
- So sánh 2 hợp đồng
- Ingest file mới
- Liệt kê document trong VectorDB
"""

import argparse
import sys
import time
import logging
from typing import Optional

from src.core.comparator import run_comparison, save_report, print_report
from src.core.llm import is_ollama_running, DEFAULT_MODEL
from src.database.vector_store import LegalVectorDB
from src.core.ingestion import extract_text
from src.core.chunking import structural_chunking


# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────
def check_ollama(model: str):
    """Check Ollama availability."""
    logger.info(f"Checking Ollama (model: {model})...")

    if not is_ollama_running(model):
        raise RuntimeError(
            f"Ollama not running. Please run: ollama run {model}"
        )

    logger.info("Ollama is ready.")


def list_docs(db_path: str):
    """List doc_ids in VectorDB (safe version)."""
    db = LegalVectorDB(db_name=db_path)

    try:
        doc_ids = db.list_doc_ids()  # ✅ bạn cần implement function này trong DB
    except AttributeError:
        logger.error("VectorDB chưa hỗ trợ list_doc_ids()")
        return

    if not doc_ids:
        logger.info("No documents found.")
        return

    logger.info("Documents in DB:")
    for doc_id, count in doc_ids.items():
        logger.info(f" - {doc_id} ({count} chunks)")


def build_metadata(chunk, doc_id: str, index: int, file_path: str):
    """Centralized metadata builder."""
    return {
        "doc_id": doc_id,
        "article": chunk.get("article", ""),
        "clause_index": index,
        "source_file": file_path,
    }


def ingest_file(file_path: str, doc_id: str, db_path: str):
    """Ingest document into VectorDB."""
    logger.info(f"Ingesting file: {file_path}")
    logger.info(f"Doc ID: {doc_id}")

    start = time.time()

    text = extract_text(file_path)
    if not text.strip():
        raise ValueError("Empty text extracted from file")

    chunks = structural_chunking(text, doc_id=doc_id)
    logger.info(f"Chunked into {len(chunks)} clauses")

    db = LegalVectorDB(db_name=db_path)

    metadata_list = [
        build_metadata(chunk, doc_id, i, file_path)
        for i, chunk in enumerate(chunks)
    ]

    db.insert_legal_chunks(chunks, metadata_list, doc_id_to_clear=doc_id)

    logger.info(f"Ingest completed in {time.time() - start:.2f}s")


# ─────────────────────────────────────────────
# Main logic
# ─────────────────────────────────────────────
def run_compare_flow(
    doc_a: str,
    doc_b: str,
    db_path: str,
    model: str,
    output: Optional[str],
    skip_check: bool
):
    """Main comparison flow."""
    if not skip_check:
        check_ollama(model)

    logger.info(f"Comparing: {doc_a} vs {doc_b}")
    logger.info(f"Model: {model}")

    start = time.time()

    results = run_comparison(
        doc_id_a=doc_a,
        doc_id_b=doc_b,
        db_path=db_path,
        model=model,
        verbose=True,
    )

    print_report(results)

    output_path = output or f"comparison_{doc_a}_vs_{doc_b}.json"
    save_report(results, output_path)

    logger.info(f"Saved report → {output_path}")
    logger.info(f"Done in {time.time() - start:.2f}s")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Legal Document Comparator CLI"
    )

    parser.add_argument("--doc_a", type=str)
    parser.add_argument("--doc_b", type=str)
    parser.add_argument("--db", type=str, default="legal_data.json")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=str)

    parser.add_argument("--list_docs", action="store_true")
    parser.add_argument("--ingest", type=str)
    parser.add_argument("--doc_id", type=str)
    parser.add_argument("--no_check", action="store_true")

    args = parser.parse_args()

    try:
        # ── LIST DOCS
        if args.list_docs:
            list_docs(args.db)
            return

        # ── INGEST
        if args.ingest:
            if not args.doc_id:
                raise ValueError("Missing --doc_id when ingesting")

            ingest_file(args.ingest, args.doc_id, args.db)
            return

        # ── COMPARE
        if not args.doc_a or not args.doc_b:
            parser.print_help()
            raise ValueError("Missing --doc_a or --doc_b")

        run_compare_flow(
            doc_a=args.doc_a,
            doc_b=args.doc_b,
            db_path=args.db,
            model=args.model,
            output=args.output,
            skip_check=args.no_check
        )

    except Exception as e:
        logger.error(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()