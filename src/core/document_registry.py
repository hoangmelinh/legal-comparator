import json
import os
from pathlib import Path
from typing import Any

DEFAULT_CACHE_ROOT = Path("data/cache")
DEFAULT_REGISTRY_PATH = DEFAULT_CACHE_ROOT / "document_registry.json"


class DocumentRegistry:
    def __init__(self, registry_path: Path | str = DEFAULT_REGISTRY_PATH):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            return {"documents": {}, "hashes": {}}

        try:
            with self.registry_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except (json.JSONDecodeError, OSError):
            loaded = {}

        loaded.setdefault("documents", {})
        loaded.setdefault("hashes", {})
        loaded["documents"] = {
            doc_id: self._sanitize_metadata(metadata or {})
            for doc_id, metadata in loaded["documents"].items()
        }
        loaded["hashes"] = {
            file_hash: self._sanitize_metadata(metadata or {})
            for file_hash, metadata in loaded["hashes"].items()
        }
        return loaded

    def save(self) -> None:
        tmp_path = self.registry_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(self.data, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.registry_path)

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        return self.data["documents"].get(doc_id)

    def get_hash_entry(self, file_hash: str) -> dict[str, Any] | None:
        return self.data["hashes"].get(file_hash)

    def _sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        sanitized = dict(metadata)
        sanitized.pop("text", None)
        sanitized.pop("records", None)
        return sanitized

    def upsert_document(self, doc_id: str, metadata: dict[str, Any]) -> None:
        current = self.data["documents"].get(doc_id, {})
        current.update(self._sanitize_metadata(metadata))
        self.data["documents"][doc_id] = current
        self.save()

    def upsert_hash_entry(self, file_hash: str, metadata: dict[str, Any]) -> None:
        current = self.data["hashes"].get(file_hash, {})
        current.update(self._sanitize_metadata(metadata))
        self.data["hashes"][file_hash] = current
        self.save()

    def find_doc_id_by_hash(self, file_hash: str) -> str | None:
        hash_entry = self.get_hash_entry(file_hash) or {}
        preferred_doc_id = hash_entry.get("canonical_doc_id")
        if preferred_doc_id:
            return preferred_doc_id

        for doc_id, metadata in self.data["documents"].items():
            if metadata.get("source_hash") == file_hash:
                return doc_id
        return None
