from nano_vectordb import NanoVectorDB
from src.models.embedding import EmbeddingModel


class LegalVectorDB:
    def __init__(self, db_name="legal_data.json"):
        """
        Khởi tạo Vector Database cho hệ thống pháp lý.
        """

        self.embedder = EmbeddingModel()

        self.db = NanoVectorDB(
            embedding_dim=1024,
            metric="cosine",
            storage_file=db_name
        )

    def insert_legal_chunks(self, chunks, metadata_list, doc_id_to_clear: str = None):
        
        doc_ids = set([meta.get('doc_id') for meta in metadata_list if 'doc_id' in meta])
        if doc_id_to_clear:
            doc_ids.add(doc_id_to_clear)

        if doc_ids:
            storage = self.db._NanoVectorDB__storage
            all_items = storage.get("data", [])

            ids_to_delete = []
            for item in all_items:
                if item.get("metadata", {}).get("doc_id") in doc_ids:
                    ids_to_delete.append(item.get("__id__"))

            if ids_to_delete:
                self.db.delete(ids_to_delete)
                print(f"--- Đã xóa {len(ids_to_delete)} chunk cũ của {list(doc_ids)} trước khi ingest ---")

        texts_to_embed = [c["content"] if isinstance(c, dict) else str(c) for c in chunks]

        if hasattr(self.embedder.model, "encode"):
            embeddings = self.embedder.model.encode(texts_to_embed, normalize_embeddings=True)
        else:
            embeddings = self.embedder.get_embeddings(texts_to_embed)

        data_to_upsert = []

        for chunk, vector, metadata in zip(chunks, embeddings, metadata_list):
            data_to_upsert.append({
                "__id__": f"{metadata['doc_id']}_{metadata['clause_index']}",
                "__vector__": vector,
                "content": chunk,
                "metadata": metadata
            })

        self.db.upsert(data_to_upsert)

        if hasattr(self.db, "persist"):
            self.db.persist()
        elif hasattr(self.db, "save"):
            self.db.save()

        print(f"--- Đã lưu {len(data_to_upsert)} điều khoản vào NanoVectorDB ---")

    def search(self, query: str, top_k: int = 5):

        if not query.strip():
            print("Query rỗng.")
            return []

        query_vector = self.embedder.get_embeddings([query])[0]
        results = self.db.query(query=query_vector, top_k=top_k)

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


    def list_doc_ids(self):
        """
        Trả về danh sách doc_id và số lượng chunk:
        {
            "HD_A": 25,
            "HD_B": 30
        }
        """
        try:
            storage = self.db._NanoVectorDB__storage
            data = storage.get("data", [])
        except AttributeError:
            return {}

        result = {}

        for item in data:
            doc_id = item.get("metadata", {}).get("doc_id", "unknown")
            result[doc_id] = result.get(doc_id, 0) + 1

        return result