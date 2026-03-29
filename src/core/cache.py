import json
import hashlib
import os

CACHE_FILE = "llm_cache.json"

class LLMCache:
    def __init__(self, cache_file=CACHE_FILE):
        self.cache_file = cache_file
        self.cache = self._load()

    def _load(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except:
                    return {}
        return {}

    def _save(self):
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def get(self, prompt: str, model: str) -> dict:
        key = hashlib.md5(f"{model}_{prompt}".encode("utf-8")).hexdigest()
        return self.cache.get(key)

    def set(self, prompt: str, model: str, result: dict):
        key = hashlib.md5(f"{model}_{prompt}".encode("utf-8")).hexdigest()
        self.cache[key] = result
        self._save()

# Global cache instance
cache = LLMCache()
