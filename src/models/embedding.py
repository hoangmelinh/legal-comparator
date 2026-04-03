import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer
class EmbeddingModel:
    def __init__(self, model_name='BAAI/bge-m3'):
        # Khởi tạo mô hình BGE-M3
        self.model = SentenceTransformer(model_name)
        print(f"[Embedding] Loaded model: {model_name}")

    def get_embeddings(self, text_list):
        """Biến danh sách các Điều/Khoản thành Vector."""
        return self.model.encode(text_list)