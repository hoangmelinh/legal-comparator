import os
from src.core.ingestion import extract_text
from src.core.normalization import clean_legal_text # Nếu bạn đã viết
from src.core.chunking import structural_chunking
from src.database.vector_store import LegalVectorDB

def run_pipeline(file_name):
    print(f"Bắt đầu xử lý: {file_name}")

    # Bước 1: Ingestion (Trích xuất text) 
    input_path = os.path.join("data", "raw", file_name)
    raw_text = extract_text(input_path)
    
    # Bước 2: Chunking (Tách theo điều khoản) [cite: 45, 117]
    # (Có thể gọi thêm clean_legal_text ở đây nếu cần)
    raw_text = clean_legal_text(raw_text)
    clauses = structural_chunking(raw_text)
    print(f"Đã tách thành {len(clauses)} điều khoản.")

    # Bước 3: Embedding & Storage (Lưu Nanovector) [cite: 11, 130]
    db = LegalVectorDB(db_name="legal_data.json")
    
    # Tạo metadata giả định để test trích dẫn sau này [cite: 47, 69]
    metadata_list = [{"doc_id": "HD_TEST_01", "clause_index": i} for i in range(len(clauses))]
    
    db.insert_legal_chunks(clauses, metadata_list)
    print("Hoàn thành! Dữ liệu đã sẵn sàng trong Nanovector DB.")

    db_path = os.path.abspath("legal_data.json")
    print("DB expected at:", db_path)
    print("File tồn tại không?", os.path.exists(db_path))

if __name__ == "__main__":
    #ten file cần test
    target_file = "hop_dong_mau_1.docx" 
    run_pipeline(target_file)