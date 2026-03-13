import os
import sys

# Đảm bảo in unicode ra console Windows không bị lỗi
sys.stdout.reconfigure(encoding='utf-8')

from src.database.vector_store import LegalVectorDB
from main import run_pipeline

def test_search():
    db_path = "legal_data.json"
    
    # Xoá DB cũ để test lại logit lọc trùng (deduplication)
    import os
    if os.path.exists(db_path):
        os.remove(db_path)
        print("Đã xoá DB cũ để test luồng Index mới.")
        
    # 1. Đảm bảo dữ liệu đã được index / persisted
    if not os.path.exists(db_path):
        print(f"Không tìm thấy persisted DB tại '{db_path}'.")
        print("Đang khởi chạy pipeline để index dữ liệu...")
        
        target_file = "hop_dong_mau_1.docx"
        raw_path = os.path.join("data", "raw", target_file)
        
        if os.path.exists(raw_path):
            run_pipeline(target_file)
        else:
            print(f"Lỗi: Không tìm thấy file mẫu '{target_file}' trong data/raw/ để index.")
            print("Vui lòng thêm file mẫu vào data/raw/ và chạy lại.")
            return
    else:
        print(f"Đã tìm thấy persisted vector database: {db_path}. Tiến hành load dữ liệu...")

    # 2. Load Vector Store
    db = LegalVectorDB(db_name=db_path)
    
    # 3. Chạy search thử nghiệm bằng query CÓ THẬT trong hợp đồng
    query = "trách nhiệm bồi thường thiệt hại phát sinh do vi phạm nghĩa vụ"
    top_k = 3
    print(f"\n────────────────────────────────────────")
    print(f"Query: {query}")
    print(f"Top K: {top_k}")
    print(f"────────────────────────────────────────\n")
    
    results = db.search(query=query, top_k=top_k)
    
    if not results:
        print("Không tìm thấy kết quả nào.")
        return

    # 4. In ra màn hình console kết quả demo
    for i, res in enumerate(results, 1):
        score = res.get("score", 0.0)
        metadata = res.get("metadata", {})
        doc_id = metadata.get("doc_id", "N/A")
        clause = metadata.get("clause_index", "N/A")
        text = res.get("text", "")
        
        # Xử lý trường hợp chunk được lưu dưới dạng dict hoặc chuỗi báo dict
        if isinstance(text, dict):
            text = text.get("content", str(text))
        elif isinstance(text, str) and text.strip().startswith("{"):
            import ast
            try:
                text_dict = ast.literal_eval(text.strip())
                if isinstance(text_dict, dict):
                    text = text_dict.get("content", text)
            except Exception:
                pass
        
        # Cắt bớt text nếu quá dài cho demo
        short_text = text[:300] + "..." if len(text) > 300 else text

        print(f"Result {i}")
        print(f"Score: {score:.4f}")
        print(f"Clause: Điều khoản số {clause} (Doc: {doc_id})")
        print(f"Text: {short_text}")
        print("-" * 40)

if __name__ == "__main__":
    test_search()
