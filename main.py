import os
from uuid import uuid4

from src.core.ingestion import ingest_document


def run_pipeline(file_name):
    print(f"Bắt đầu xử lý: {file_name}")

    input_path = os.path.join("data", "raw", file_name)
    doc_id = f"run_{uuid4().hex[:12]}"
    result = ingest_document(input_path, doc_id=doc_id, db_path="legal_data.json")

    print("Hoàn thành! Dữ liệu đã sẵn sàng trong NanoVectorDB.")
    print("Normalized PDF:", result["normalized_pdf_path"])
    print("Conversion method:", result["conversion_method"])
    print("Ingest status:", result["ingest_status"])
    if result.get("warning"):
        print("Warning:", result["warning"])

    db_path = os.path.abspath("legal_data.json")
    print("DB expected at:", db_path)
    print("File tồn tại không?", os.path.exists(db_path))


if __name__ == "__main__":
    target_file = "hop_dong_mau_1.docx"
    run_pipeline(target_file)
