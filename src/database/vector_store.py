from nano_vectordb import NanoVectorDB
from src.models.embedding import EmbeddingModel


class LegalVectorDB:
    def __init__(self, db_name="legal_data.json"):
        # Load embedding model
        self.embedder = EmbeddingModel()

        # Init NanoVectorDB (bge-m3 = 1024 dim)
        self.db = NanoVectorDB(
            embedding_dim=1024,
            metric="cosine",
            storage_file=db_name
        )

    def insert_legal_chunks(self, chunks, metadata_list):

        if not chunks:
            print("⚠ Không có chunk nào để lưu.")
            return

        # 1️⃣ Tạo embeddings
        embeddings = self.embedder.get_embeddings(chunks)

        data_to_upsert = []

        for i, (chunk, vector, metadata) in enumerate(
            zip(chunks, embeddings, metadata_list)
        ):

            # Đảm bảo vector là numpy array
            # KHÔNG convert sang list
            # NanoVectorDB yêu cầu ndarray để hash

            data_to_upsert.append({
                "__id__": f"{metadata['doc_id']}_{metadata['clause_index']}",
                "__vector__": vector,   # giữ nguyên numpy array
                "content": chunk,
                "metadata": metadata
            })

        # Upsert vào DB
        self.db.upsert(data_to_upsert)

        if hasattr(self.db, "persist"):
            self.db.persist()
        elif hasattr(self.db, "save"):
            self.db.save()

        print(f"--- Đã lưu {len(data_to_upsert)} điều khoản vào Nanovector ---")