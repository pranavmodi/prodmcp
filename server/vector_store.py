import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np


def _list_json_files(data_dir: str) -> List[Path]:
    root = Path(data_dir)
    if not root.exists():
        return []
    files: List[Path] = []
    files.extend(root.glob("*.json"))
    for sub in root.rglob("*.json"):
        if sub not in files:
            files.append(sub)
    return files


class SimpleVectorizer:
    """A minimal embedding stub using hashing trick. Replace with real embeddings later."""
    def __init__(self, dim: int = 768, seed: int = 42):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.token_cache: Dict[str, np.ndarray] = {}

    def _token_to_vec(self, token: str) -> np.ndarray:
        if token in self.token_cache:
            return self.token_cache[token]
        h = abs(hash(token))
        self.rng = np.random.default_rng(h % (2**32))
        v = self.rng.standard_normal(self.dim).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-8)
        self.token_cache[token] = v
        return v

    def embed_text(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(self.dim, dtype=np.float32)
        tokens = [t for t in text.lower().split() if t]
        if not tokens:
            return np.zeros(self.dim, dtype=np.float32)
        vecs = [self._token_to_vec(t) for t in tokens[:512]]
        v = np.mean(vecs, axis=0)
        v /= (np.linalg.norm(v) + 1e-8)
        return v.astype(np.float32)


class OpenAIEmbedder:
    """Optional OpenAI embedding backend for higher-quality vectors."""
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self._client = None
        try:
            # Lazy import to avoid hard dependency
            from openai import OpenAI  # type: ignore
            self._client = OpenAI()
        except Exception:
            self._client = None

    def is_available(self) -> bool:
        return self._client is not None and bool(os.getenv("OPENAI_API_KEY"))

    def embed_texts(self, texts: List[str]) -> Optional[np.ndarray]:
        try:
            if not self.is_available():
                return None
            if not texts:
                return np.zeros((0, 1536), dtype=np.float32)
            # The SDK accepts a list input; returns data[].embedding
            resp = self._client.embeddings.create(model=self.model, input=texts)
            vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
            arr = np.vstack(vecs)
            # Normalize for cosine/inner-product
            import faiss  # type: ignore
            faiss.normalize_L2(arr)
            return arr
        except Exception as e:
            try:
                import logging as _logging
                _logging.getLogger(__name__).error(f"OpenAI Embeddings call failed: {e}", exc_info=True)
            except Exception:
                pass
            return None


class FAISSStore:
    def __init__(self, tenant_id: str, dim: int = 768):
        self.tenant_id = tenant_id
        self.dim = dim
        self.index = None
        self.meta: List[Dict[str, Any]] = []
        self._vectorizer = SimpleVectorizer(dim=dim)
        self._embedder = OpenAIEmbedder(model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))

        # JSON data lives per-tenant under DATA_DIR/<tenant_id>
        data_root = Path(os.getenv("DATA_DIR", "./scraped_pages"))
        self.data_dir = data_root / tenant_id
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Store FAISS artifacts under DATA_DIR/faiss_data/<tenant_id>
        faiss_dir = data_root / "faiss_data" / tenant_id
        faiss_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = faiss_dir / "faiss.index"
        self.meta_path = faiss_dir / "faiss.meta.json"

    def _load_index(self) -> bool:
        try:
            import faiss  # type: ignore
        except Exception:
            return False
        if not self.index_path.exists() or not self.meta_path.exists():
            return False
        try:
            self.index = faiss.read_index(str(self.index_path))
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
            return True
        except Exception:
            return False

    def _save_index(self) -> None:
        try:
            import faiss  # type: ignore
        except Exception:
            return
        if self.index is None:
            return
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f)

    def build(self, max_chars: int = 1000, overlap: int = 150) -> int:
        try:
            import faiss  # type: ignore
        except Exception:
            return 0

        chunks: List[str] = []
        meta: List[Dict[str, Any]] = []
        for fp in _list_json_files(str(self.data_dir)):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Prefer markdown (has paragraph breaks); fallback to content
                content = (data.get("markdown") or data.get("content") or "").strip()
                url = data.get("url") or str(fp)
                title = data.get("title") or ""
                if not content:
                    continue
                paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                if not paragraphs:
                    # Fallback split by single newlines or sentences if no double-newline blocks
                    tmp = [p.strip() for p in content.split("\n") if p.strip()]
                    if len(tmp) > 1:
                        paragraphs = tmp
                    else:
                        # Rough sentence split
                        paragraphs = [s.strip() for s in content.replace("? ", "?\n").replace(". ", ".\n").split("\n") if s.strip()]
                current = ""
                for p in paragraphs or [content]:
                    if len(current) + len(p) + 2 <= max_chars:
                        current = (current + "\n\n" + p) if current else p
                    else:
                        if current:
                            chunks.append(current)
                            meta.append({"url": url, "title": title, "snippet": current[:1200]})
                        if len(p) > max_chars:
                            start = 0
                            while start < len(p):
                                end = min(len(p), start + max_chars)
                                chunks.append(p[start:end])
                                meta.append({"url": url, "title": title, "snippet": p[start:end][:1200]})
                                start = max(start + max_chars - overlap, end)
                            current = ""
                        else:
                            current = p
                if current:
                    chunks.append(current)
                    meta.append({"url": url, "title": title, "snippet": current[:1200]})
            except Exception:
                continue

        if not chunks:
            return 0

        # Try OpenAI embeddings first (if available), else fallback to hashing vectorizer
        embeddings = None
        if self._embedder.is_available():
            embeddings = self._embedder.embed_texts(chunks)
        if embeddings is None:
            embeddings = np.vstack([self._vectorizer.embed_text(t) for t in chunks])
        try:
            import faiss  # type: ignore
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(self.dim)
            index.add(embeddings)
        except Exception:
            return 0
        self.index = index
        self.meta = meta
        self._save_index()
        return len(chunks)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None and not self._load_index():
            return []
        try:
            import faiss  # type: ignore
        except Exception:
            return []
        # Use OpenAI embeddings if available for query; fallback to hashing
        if self._embedder.is_available():
            emb = self._embedder.embed_texts([query])
            if emb is None or emb.shape[0] == 0:
                q = self._vectorizer.embed_text(query).reshape(1, -1)
            else:
                q = emb.reshape(1, -1)
        else:
            q = self._vectorizer.embed_text(query).reshape(1, -1)
        try:
            import faiss  # type: ignore
            faiss.normalize_L2(q)
            D, I = self.index.search(q, k)
        except Exception:
            return []
        results: List[Dict[str, Any]] = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.meta):
                continue
            results.append({
                "score": float(score),
                "url": self.meta[idx].get("url"),
                "title": self.meta[idx].get("title", ""),
                "snippet": self.meta[idx].get("snippet", ""),
            })
        return results

    def embed_text(self, text: str) -> np.ndarray:
        """Expose a single-text embedding using best available backend."""
        if self._embedder.is_available():
            arr = self._embedder.embed_texts([text])
            if arr is not None and arr.shape[0] > 0:
                return arr[0]
        return self._vectorizer.embed_text(text)