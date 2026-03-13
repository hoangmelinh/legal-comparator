from nano_vectordb import NanoVectorDB
from src.models.embedding import EmbeddingModel


class LegalVectorDB:
    def __init__(self, db_name="legal_data.json"):
        """
        Khởi tạo Vector Database cho hệ thống pháp lý.
        """

        # Load embedding model
        self.embedder = EmbeddingModel()

        # Init NanoVectorDB (bge-m3 = 1024 dimensions)
        self.db = NanoVectorDB(
            embedding_dim=1024,
            metric="cosine",
            storage_file=db_name
        )

    def insert_legal_chunks(self, chunks, metadata_list):
        """
        Lưu các điều khoản pháp lý (chunks) vào Vector DB.
        """

        # 1. Tạo embeddings
        # Đảm bảo chỉ truyền list of strings vào model thay vì list of dicts
        texts_to_embed = [c["content"] if isinstance(c, dict) else str(c) for c in chunks]
        
        # BGE-M3 nên dùng normalize_embeddings=True để Cosine Similarity hoạt động chuẩn xác
        if hasattr(self.embedder.model, "encode"):
            embeddings = self.embedder.model.encode(texts_to_embed, normalize_embeddings=True)
        else:
            embeddings = self.embedder.get_embeddings(texts_to_embed)

        data_to_upsert = []

        for chunk, vector, metadata in zip(chunks, embeddings, metadata_list):

            data_to_upsert.append({
                "__id__": f"{metadata['doc_id']}_{metadata['clause_index']}",
                "__vector__": vector,  # giữ nguyên numpy array
                "content": chunk,
                "metadata": metadata
            })

        # 2. Upsert vào Vector DB
        self.db.upsert(data_to_upsert)

        # 3. Persist xuống disk
        if hasattr(self.db, "persist"):
            self.db.persist()
        elif hasattr(self.db, "save"):
            self.db.save()

        print(f"--- Đã lưu {len(data_to_upsert)} điều khoản vào NanoVectorDB ---")

    def search(self, query: str, top_k: int = 5):
        """
        Tìm kiếm các điều khoản liên quan nhất tới query.

        Return:
        [
            {
                "text": "...",
                "metadata": {...},
                "score": 0.75
            }
        ]
        """

        if not query.strip():
            print("Query rỗng.")
            return []

        # 1. Query -> Embedding
        query_vector = self.embedder.get_embeddings([query])[0]

        # 2. Vector Search
        results = self.db.query(query=query_vector, top_k=top_k)

        # 3. Format kết quả
        formatted = []

        for r in results:

            score = r.get("__metrics__", 0.0)

            if isinstance(score, list):
                score = score[0]

            formatted.append({
                "text": r.get("content", ""),
                "metadata": r.get("metadata", {}),
                "score": float(score)
            })

        return formatted