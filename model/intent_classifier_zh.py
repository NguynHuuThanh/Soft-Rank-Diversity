"""
Chinese intent classifier for TG-ReDial.

Replaces the CR-Walker-style human-labeled intents (which only exist for
ReDial) with predictions from a pretrained multilingual NLI model. The
underlying model (`joeddav/xlm-roberta-large-xnli`) supports zero-shot
classification in Chinese, so we can label each system turn at preprocess
time without any task-specific fine-tuning.

The mapping is fixed to the 3 CR-Walker intent classes that the rest of
the pipeline expects: {"chat", "question", "recommend"}.

A rule-based fallback derives the intent directly from the dataset's
existing `target`/`context_policy` field (which already carries Chinese
policy labels: 谈论 / 请求推荐 / 允许推荐 / 拒绝). The fallback runs
when the model cannot be loaded (e.g. offline) or when the user passes
`use_model=False`. Predictions are cached on disk so preprocessing is
deterministic and cheap to re-run.
"""

import hashlib
import json
import os
import os.path as osp
from typing import Iterable, List, Optional


# Map the 4 native TG-ReDial policy labels onto the 3 CR-Walker classes.
# 谈论       -> chat        (free chat / discussing something)
# 请求推荐   -> question    (the system asks for the user's preference)
# 允许推荐   -> recommend   (the system makes the recommendation)
# 拒绝       -> chat        (rejecting / declining is treated as chat)
NATIVE_LABEL_MAP = {
    "谈论": "chat",
    "请求推荐": "question",
    "允许推荐": "recommend",
    "拒绝": "chat",
}

# Zero-shot candidate labels in Chinese (kept aligned with CR-Walker classes).
ZH_CANDIDATES = ["闲聊", "提问", "推荐"]
ZH_TO_EN = {"闲聊": "chat", "提问": "question", "推荐": "recommend"}

DEFAULT_MODEL = "joeddav/xlm-roberta-large-xnli"


class ChineseIntentClassifier:
    """Zero-shot Chinese intent classifier with caching + rule fallback."""

    def __init__(
        self,
        cache_path: Optional[str] = None,
        model_name: str = DEFAULT_MODEL,
        use_model: bool = True,
        device: int = -1,
    ):
        self.cache_path = cache_path
        self.model_name = model_name
        self.use_model = use_model
        self.device = device
        self._pipe = None
        self._cache = {}

        if cache_path and osp.isfile(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

    # ---------------------------------------------------------------- model
    def _ensure_model(self) -> bool:
        if self._pipe is not None:
            return True
        if not self.use_model:
            return False
        try:
            from transformers import pipeline  # type: ignore

            self._pipe = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=self.device,
            )
            return True
        except Exception as e:
            print(f"[intent_zh] Falling back to rule-based labels — pipeline init failed: {e}")
            self.use_model = False
            return False

    # ----------------------------------------------------------------- util
    @staticmethod
    def _key(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    @staticmethod
    def from_native_policy(policy_label: Optional[str]) -> Optional[str]:
        """Map a raw TG-ReDial policy label to a CR-Walker intent, or None."""
        if not policy_label:
            return None
        return NATIVE_LABEL_MAP.get(policy_label)

    # -------------------------------------------------------------- predict
    def predict(self, text: str, native_hint: Optional[str] = None) -> str:
        """Return one of {"chat", "question", "recommend"}.

        If `native_hint` is provided (e.g. the policy label that already ships
        with the dataset), the rule-based mapping wins — this keeps the
        labelling consistent with how the dataset authors annotated the turn
        and avoids drifting on samples the model is uncertain about.
        """
        mapped = self.from_native_policy(native_hint)
        if mapped is not None:
            return mapped

        if not text:
            return "chat"

        key = self._key(text)
        if key in self._cache:
            return self._cache[key]

        if self._ensure_model():
            try:
                out = self._pipe(text, candidate_labels=ZH_CANDIDATES)
                top = out["labels"][0]
                label = ZH_TO_EN.get(top, "chat")
            except Exception as e:
                print(f"[intent_zh] inference failed ({e}); defaulting to chat")
                label = "chat"
        else:
            label = "chat"

        self._cache[key] = label
        return label

    def predict_batch(
        self,
        texts: Iterable[str],
        native_hints: Optional[Iterable[Optional[str]]] = None,
    ) -> List[str]:
        hints = list(native_hints) if native_hints is not None else [None] * len(list(texts))
        # texts was consumed by len(); rematerialise
        texts = list(texts) if not isinstance(texts, list) else texts
        return [self.predict(t, h) for t, h in zip(texts, hints)]

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        os.makedirs(osp.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False)


if __name__ == "__main__":
    clf = ChineseIntentClassifier(use_model=False)
    samples = [
        ("你最近喜欢看什么电影？", "请求推荐"),
        ("我推荐《让子弹飞》给你。", "允许推荐"),
        ("今天天气真不错。", "谈论"),
    ]
    for txt, hint in samples:
        print(txt, "->", clf.predict(txt, native_hint=hint))
