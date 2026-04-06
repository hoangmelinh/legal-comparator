import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer
class EmbeddingModel:
    def __init__(self, model_name='BAAI/bge-m3'):
        try:
            # Khởi tạo mô hình BGE-M3 từ local cache
            self.model = SentenceTransformer(model_name)
            print(f"[Embedding] Loaded model: {model_name} (Offline Mode)")
        except Exception as e:
            raise RuntimeError(
                f"\n[LỖI OFFLINE] Không tìm thấy model '{model_name}' trong cache Hugging Face ở máy tính.\n"
                f"Do hệ thống đang ép chạy offline (100% không kết nối mạng), bạn CẦN PHẢI:\n"
                f" 1. Kết nối internet lại.\n"
                f" 2. Chạy ứng dụng một lần để tải model về cache.\n"
                f" 3. Sau đó mới ngắt mạng để sử dụng hoàn toàn nội bộ.\n"
                f"Chi tiết lỗi: {e}"
            )

    def get_embeddings(self, text_list):
        """Biến danh sách các Điều/Khoản thành Vector."""
        return self.model.encode(text_list)