import hashlib
import json
import os
import threading

CACHE_FILE = "llm_cache.json"


class LLMCache:
    def __init__(self, cache_file=CACHE_FILE):
        self.cache_file = cache_file
        self._lock = threading.Lock()
        self.cache = self._load()

    def _load(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, encoding="utf-8") as f:
                try:
                    return json.load(f)
                except:
                    return {}
        return {}

    def _save(self):
        tmp_file = f"{self.cache_file}.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp_file, self.cache_file)

    def get(self, prompt: str, model: str) -> dict:
        key = hashlib.md5(f"{model}_{prompt}".encode()).hexdigest()
        with self._lock:
            return self.cache.get(key)

    def set(self, prompt: str, model: str, result: dict):
        key = hashlib.md5(f"{model}_{prompt}".encode()).hexdigest()
        with self._lock:
            self.cache[key] = result
            self._save()


cache = LLMCache()
