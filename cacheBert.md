# cacheBert

This document provides a **single-file implementation** for optimizing BERT usage during training.

It implements:
- **Process-level cache** for `BertModel` + `BertTokenizer` (loaded once, reused)
- **Optional BERT freezing** (`requires_grad=False`) to reduce compute and memory
- **`eval()` mode on BERT** when frozen to disable dropout noise
- **Tokenization cache (LRU)** to avoid repeated tokenization for repeated dialog histories
- Compatibility with the existing `Utterance_Embedder` API shape used in this project

---

## Full Code (single file)

```python
import json
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from nltk import word_tokenize
from transformers import BertModel, BertTokenizer


class _SharedBertCache:
    """
    Process-level cache for pretrained BERT model/tokenizer.
    Keyed by pretrained model name.
    """

    _tokenizers: Dict[str, BertTokenizer] = {}
    _models: Dict[str, BertModel] = {}

    @classmethod
    def get_tokenizer(cls, pretrained_weights: str) -> BertTokenizer:
        tok = cls._tokenizers.get(pretrained_weights)
        if tok is None:
            tok = BertTokenizer.from_pretrained(pretrained_weights)
            cls._tokenizers[pretrained_weights] = tok
        return tok

    @classmethod
    def get_model(cls, pretrained_weights: str) -> BertModel:
        model = cls._models.get(pretrained_weights)
        if model is None:
            model = BertModel.from_pretrained(pretrained_weights)
            cls._models[pretrained_weights] = model
        return model


class _DialogTokenizationLRU:
    """
    LRU cache storing CPU tokenization tensors keyed by tuple of utterances.
    This avoids repeated tokenizer calls for repeated histories.
    """

    def __init__(self, max_entries: int = 4096):
        self.max_entries = max_entries
        self._cache: OrderedDict[Tuple[str, ...], Dict[str, torch.Tensor]] = OrderedDict()

    def get(self, key: Tuple[str, ...]):
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        return None

    def put(self, key: Tuple[str, ...], value: Dict[str, torch.Tensor]):
        if key in self._cache:
            self._cache.pop(key)
        self._cache[key] = value
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)


class Utterance_Embedder(nn.Module):
    def __init__(
        self,
        rnn_type: str = "RNN_TANH",
        use_bert: bool = True,
        rnn_hidden: int = 64,
        dropout: float = 0.5,
        num_turns: int = 10,
        num_words: int = 30,
        word_net: bool = False,
        pretrained_weights: str = "bert-base-uncased",
        freeze_bert: bool = True,
        share_bert_across_instances: bool = True,
        token_cache_size: int = 4096,
    ):
        super(Utterance_Embedder, self).__init__()

        self.rnn_hidden = rnn_hidden
        self.num_turns = num_turns
        self.num_words = num_words
        self.word_net = word_net
        self.use_bert = use_bert
        self.freeze_bert = freeze_bert

        if share_bert_across_instances:
            self.tokenizer = _SharedBertCache.get_tokenizer(pretrained_weights)
            self.model = _SharedBertCache.get_model(pretrained_weights)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
            self.model = BertModel.from_pretrained(pretrained_weights)

        if self.freeze_bert:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        if rnn_type in ["LSTM", "GRU"]:
            self.rnn = getattr(nn, rnn_type)(768, rnn_hidden, dropout=dropout)
        else:
            try:
                nonlinearity = {"RNN_TANH": "tanh", "RNN_RELU": "relu"}[rnn_type]
            except KeyError as error:
                raise ValueError(
                    "An invalid option for rnn_type was supplied, options are "
                    "['LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU']"
                ) from error
            self.rnn = nn.RNN(768, rnn_hidden, nonlinearity=nonlinearity, dropout=dropout)

        self.rnn_type = rnn_type

        root = osp.dirname(osp.dirname(osp.abspath(__file__)))
        self.key2index = json.load(open(osp.join(root, "data", "key2index_3rd.json"), encoding="utf-8"))

        self._token_lru = _DialogTokenizationLRU(max_entries=token_cache_size)

    def _bert_forward(self, tokenized: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.freeze_bert:
            with torch.no_grad():
                return self.model(**tokenized).last_hidden_state
        return self.model(**tokenized).last_hidden_state

    def forward(self, tokenized, length, max_len, init_hidden):
        bert_output = self._bert_forward(tokenized)
        bert_embed = bert_output.view(-1, self.num_turns, max_len, 768).permute(1, 0, 2, 3)
        sentence_rep = bert_embed[:, :, 0, :]

        output, hidden = self.rnn(sentence_rep, init_hidden)
        output = output[-1, :, :]
        return output

    def _build_padded_history(self, dialog_history: Sequence[Sequence[str]]) -> List[str]:
        pad_history = []
        for history in dialog_history:
            turns = len(history)
            if turns > self.num_turns:
                padded = history[turns - self.num_turns :]
            elif turns < self.num_turns:
                padded = [""] * (self.num_turns - turns) + list(history)
            else:
                padded = list(history)
            pad_history.extend(padded)
        return pad_history

    def _tokenize_with_cache(self, pad_history: List[str]):
        key = tuple(pad_history)
        cached = self._token_lru.get(key)
        if cached is not None:
            return {k: v.clone() for k, v in cached.items()}

        tokenized = self.tokenizer(pad_history, padding=True, return_tensors="pt")
        self._token_lru.put(key, {k: v.cpu().clone() for k, v in tokenized.items()})
        return tokenized

    def prepare_data(self, dialog_history, device, raw_history: bool = False):
        pad_history = []
        word_index = []
        word_batch_index = []
        all_words = []

        for i in range(len(dialog_history)):
            turn_words = []
            cur_history = ""
            turns = len(dialog_history[i])

            if turns > self.num_turns:
                padded = dialog_history[i][turns - self.num_turns :]
                pad_history.extend(padded)
                for sentence in padded:
                    cur_history += sentence + " "
            elif turns < self.num_turns:
                pad = [""] * (self.num_turns - turns)
                padded = pad + dialog_history[i]
                pad_history.extend(padded)
                for sentence in dialog_history[i]:
                    cur_history += sentence + " "
            else:
                pad_history.extend(dialog_history[i])
                for sentence in dialog_history[i]:
                    cur_history += sentence + " "

            token_history = word_tokenize(cur_history)
            if len(token_history) > self.num_words:
                token_history = token_history[-self.num_words :]
            elif len(token_history) == 0:
                word_index.append(0)
                word_batch_index.append(i)

            for word in token_history:
                idx = self.key2index.get(word.lower(), 0)
                word_index.append(idx)
                word_batch_index.append(i)
                turn_words.append(idx)
            all_words.append(turn_words)

        tokenized_dialog = self._tokenize_with_cache(pad_history)
        for key in tokenized_dialog.keys():
            tokenized_dialog[key] = tokenized_dialog[key].to(device=device)

        all_length = torch.sum(tokenized_dialog["attention_mask"], dim=-1).view(-1, self.num_turns).permute(1, 0)
        maxlen = tokenized_dialog["attention_mask"].size()[-1]
        batch_size = len(dialog_history)

        if self.rnn_type == "LSTM":
            h0 = torch.zeros(1, batch_size, self.rnn_hidden, device=device)
            c0 = torch.zeros(1, batch_size, self.rnn_hidden, device=device)
            init_hidden = (h0, c0)
        else:
            init_hidden = torch.zeros(1, batch_size, self.rnn_hidden, device=device)

        w_index = None
        w_batch_index = None
        if self.word_net:
            w_index = torch.tensor(word_index, dtype=torch.long, device=device)
            w_batch_index = torch.tensor(word_batch_index, dtype=torch.long, device=device)

        if raw_history:
            return tokenized_dialog, all_length, maxlen, init_hidden, w_index, w_batch_index, all_words
        return tokenized_dialog, all_length, maxlen, init_hidden, w_index, w_batch_index


if __name__ == "__main__":
    print("Utterance_Embedder cache-ready implementation loaded.")
```

---

## How to use in your project

1. Replace your existing utterance embedder implementation with the class above (or copy into a new module and import it from `CR_walker.py`).
2. Keep `freeze_bert=True` for faster/cheaper training when you only train downstream modules.
3. If you run multiple model instances in one process, keep `share_bert_across_instances=True`.
4. Tune `token_cache_size` based on memory budget.

---

## Notes

- This solution optimizes **in-process training runtime** and repeated tokenization overhead.
- It does **not** change model outputs semantically when `freeze_bert=True` and BERT was already effectively frozen by your training script.
- If you want BERT fine-tuning, set `freeze_bert=False`.
