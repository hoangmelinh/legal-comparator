import numpy as np
from nano_vectordb import NanoVectorDB

from src.models.embedding import EmbeddingModel


class LegalVectorDB:
    def __init__(self, db_name="legal_data.json"):
        """
        Khởi tạo Vector Database cho hệ thống pháp lý.
        """
        self._embedder = None
        self.db = NanoVectorDB(
            embedding_dim=1024,
            metric="cosine",
            storage_file=db_name
        )

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = EmbeddingModel()
        return self._embedder

    def _get_storage_items(self):
        try:
            storage = self.db._NanoVectorDB__storage
        except AttributeError:
            return []
        return storage.get("data", [])

    def _persist(self):
        if hasattr(self.db, "persist"):
            self.db.persist()
        elif hasattr(self.db, "save"):
            self.db.save()

    def delete_doc_ids(self, doc_ids):
        doc_ids = {doc_id for doc_id in doc_ids if doc_id}
        if not doc_ids:
            return 0

        ids_to_delete = []
        for item in self._get_storage_items():
            if item.get("metadata", {}).get("doc_id") in doc_ids:
                ids_to_delete.append(item.get("__id__"))

        if ids_to_delete:
            self.db.delete(ids_to_delete)
            self._persist()
        return len(ids_to_delete)

    def insert_legal_chunks(self, chunks, metadata_list, doc_id_to_clear: str = None):
        doc_ids = {meta.get("doc_id") for meta in metadata_list if meta.get("doc_id")}
        if doc_id_to_clear:
            doc_ids.add(doc_id_to_clear)

        if doc_ids:
            deleted = self.delete_doc_ids(doc_ids)
            if deleted:
                print(f"--- Deleted {deleted} existing chunks for {list(doc_ids)} before ingest ---")

        if not chunks:
            return []

        texts_to_embed = [chunk["content"] if isinstance(chunk, dict) else str(chunk) for chunk in chunks]

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

        serializable_entries = [
            {
                "__id__": entry["__id__"],
                "__vector__": entry["__vector__"].tolist() if hasattr(entry["__vector__"], "tolist") else entry["__vector__"],
                "content": entry["content"],
                "metadata": entry["metadata"],
            }
            for entry in data_to_upsert
        ]

        self.db.upsert(data_to_upsert)
        self._persist()
        print(f"--- Stored {len(data_to_upsert)} clauses in NanoVectorDB ---")
        return serializable_entries

    def insert_precomputed_entries(self, entries, target_doc_id: str, doc_id_to_clear: str = None):
        if doc_id_to_clear:
            self.delete_doc_ids({doc_id_to_clear})

        prepared_entries = []
        for index, entry in enumerate(entries):
            metadata = dict(entry.get("metadata", {}))
            metadata["doc_id"] = target_doc_id
            metadata["clause_index"] = metadata.get("clause_index", index)

            prepared_entries.append({
                "__id__": f"{target_doc_id}_{metadata['clause_index']}",
                "__vector__": np.array(entry.get("__vector__")),
                "content": entry.get("content"),
                "metadata": metadata
            })

        if prepared_entries:
            self.db.upsert(prepared_entries)
            self._persist()

        print(f"--- Restored {len(prepared_entries)} cached chunks for {target_doc_id} ---")
        return prepared_entries

    def search(self, query: str, top_k: int = 5):
        if not query.strip():
            print("Query rỗng.")
            return []

        query_vector = self.embedder.get_embeddings([query])[0]
        results = self.db.query(query=query_vector, top_k=top_k)

        formatted = []
        for result in results:
            score = result.get("__metrics__", 0.0)
            if isinstance(score, list):
                score = score[0]

            formatted.append({
                "text": result.get("content", ""),
                "metadata": result.get("metadata", {}),
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
        result = {}
        for item in self._get_storage_items():
            doc_id = item.get("metadata", {}).get("doc_id", "unknown")
            result[doc_id] = result.get(doc_id, 0) + 1
        return result

    def doc_exists(self, doc_id: str) -> bool:
        return self.list_doc_ids().get(doc_id, 0) > 0
