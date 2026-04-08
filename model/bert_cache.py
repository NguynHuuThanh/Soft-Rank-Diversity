"""
On-disk cache for BERT [CLS] embeddings of dialogue turns.

CR-Walker freezes the BERT branch (`utter_embedder.model.*` is never in the
training scripts' `unfreeze_layers`), so the [CLS] vector for a given
(model_name, utterance_text) pair is deterministic across epochs. The
training loop currently re-runs BERT on every utterance every epoch — this
cache eliminates that redundant work.

Storage layout
--------------
A single ``.pt`` dict mapping ``md5(model_name + "::" + text) -> float16
tensor of shape (768,)``. The dict is loaded into RAM at construction and
flushed back to disk periodically (every `flush_every` writes) and on
explicit `flush()` calls. The model_name is mixed into the key so the
same cache file can hold both `bert-base-uncased` and `bert-base-chinese`
entries side by side without collisions.

Concurrency
-----------
Single-process training only — a `threading.Lock` guards writes so the
cache is safe to call from a DataLoader worker, but cross-process sharing
is not supported.
"""

import hashlib
import os
import threading
from typing import List, Optional, Tuple

import torch


class BertCLSCache:
    def __init__(
        self,
        path: str,
        model_name: str,
        flush_every: int = 5000,
    ):
        self.path = path
        self.model_name = model_name
        self.flush_every = flush_every
        self._lock = threading.Lock()
        self._dirty = 0

        if path and os.path.isfile(path):
            try:
                self._cache = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                self._cache = torch.load(path, map_location="cpu")
            except Exception as e:
                print(f"[bert_cache] failed to load {path}: {e}; starting fresh")
                self._cache = {}
        else:
            self._cache = {}

        print(f"[bert_cache] path={path} model={model_name} entries={len(self._cache)}")

    # ------------------------------------------------------------------ key
    def _key(self, text: str) -> str:
        return hashlib.md5(f"{self.model_name}::{text}".encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------ get
    def get_many(
        self, texts: List[str]
    ) -> Tuple[List[Optional[torch.Tensor]], List[int], List[str]]:
        """Return (hits, miss_indices, miss_texts).

        `hits[i]` is either a cached float16 tensor (768,) or None when missing.
        """
        hits: List[Optional[torch.Tensor]] = []
        miss_idx: List[int] = []
        miss_txt: List[str] = []
        for i, t in enumerate(texts):
            v = self._cache.get(self._key(t))
            hits.append(v)
            if v is None:
                miss_idx.append(i)
                miss_txt.append(t)
        return hits, miss_idx, miss_txt

    # ------------------------------------------------------------------ put
    def put_many(self, texts: List[str], vectors: List[torch.Tensor]) -> None:
        with self._lock:
            for t, v in zip(texts, vectors):
                self._cache[self._key(t)] = v.detach().to("cpu", dtype=torch.float16)
                self._dirty += 1
            if self._dirty >= self.flush_every:
                self._save_locked()
                self._dirty = 0

    # ------------------------------------------------------------------ flush
    def _save_locked(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        torch.save(self._cache, tmp)
        os.replace(tmp, self.path)

    def flush(self) -> None:
        with self._lock:
            self._save_locked()
            self._dirty = 0

    def __len__(self) -> int:
        return len(self._cache)
