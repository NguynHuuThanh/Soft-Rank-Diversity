import os
import sys
from transformers import BertModel, BertTokenizer
from nltk import word_tokenize
import torch
import torch.nn as nn
import os.path as osp
import json

from torch_geometric.loader import DataLoader

from bert_cache import BertCLSCache


root = osp.dirname(osp.dirname(osp.abspath(__file__)))
path = osp.join(root, "data", "redial")


class Utterance_Embedder(nn.Module):
    def __init__(
        self,
        rnn_type="RNN_TANH",
        use_bert=True,
        rnn_hidden=64,
        dropout=0.5,
        num_turns=10,
        num_words=30,
        word_net=False,
        pretrained_weights=None,
        cache_path=None,
    ):
        super(Utterance_Embedder, self).__init__()
        # Allow overrides via env vars so train scripts don't need new flags.
        pretrained_weights = (
            pretrained_weights
            or os.environ.get("BERT_MODEL_NAME")
            or "bert-base-uncased"
        )
        # Predetermined per-model cache file under <root>/saved/. Always on:
        # if the file exists we reuse it, otherwise we fill it lazily on the
        # first epoch. The model name is part of the filename so different
        # checkpoints (e.g. bert-base-uncased vs bert-base-chinese) and
        # different datasets that share a checkpoint coexist safely.
        if cache_path is None:
            cache_path = os.environ.get("BERT_CACHE_PATH")
        if cache_path is None:
            safe = pretrained_weights.replace("/", "_")
            cache_path = osp.join(root, "saved", f"bert_cls_{safe}.pt")

        model_class = BertModel
        self.pretrained_weights = pretrained_weights
        self.rnn_hidden = rnn_hidden
        self.num_turns = num_turns
        self.num_words = num_words
        self.word_net = word_net
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        self.model = model_class.from_pretrained(pretrained_weights)
        # BERT is frozen across all train scripts (not in unfreeze_layers).
        # Make it explicit so cached forward passes are obviously safe and
        # so any accidental fine-tuning regressions are caught loudly.
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        self.cache = BertCLSCache(cache_path, pretrained_weights)
        self._pad_history_buf = None  # populated by prepare_data, consumed by forward

        if rnn_type in ["LSTM", "GRU"]:
            self.rnn = getattr(nn, rnn_type)(768, rnn_hidden, dropout=dropout)
        else:
            try:
                nonlinearity = {"RNN_TANH": "tanh", "RNN_RELU": "relu"}[rnn_type]
            except KeyError:
                raise ValueError(
                    """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']"""
                )
            self.rnn = nn.RNN(
                768, rnn_hidden, nonlinearity=nonlinearity, dropout=dropout
            )

        self.rnn_type = rnn_type
        self.key2index = json.load(
            open(osp.join(root, "data", "key2index_3rd.json"), encoding="utf-8")
        )

    # --------------------------------------------------------------- cached
    def _encode_cls_cached(self, texts):
        """Return (B*T, 768) float32 CLS vectors using the on-disk cache."""
        device = next(self.model.parameters()).device
        hits, miss_idx, miss_txt = self.cache.get_many(texts)

        miss_vecs = []
        if miss_txt:
            with torch.no_grad():
                tok = self.tokenizer(
                    miss_txt,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                tok = {k: v.to(device) for k, v in tok.items()}
                cls = self.model(**tok)[0][:, 0, :]  # (M, 768)
            miss_vecs = [cls[i].detach() for i in range(cls.size(0))]
            self.cache.put_many(miss_txt, miss_vecs)

        out = []
        miss_iter = iter(miss_vecs)
        for h in hits:
            if h is None:
                v = next(miss_iter).to(device=device, dtype=torch.float32)
            else:
                v = h.to(device=device, dtype=torch.float32)
            out.append(v)
        return torch.stack(out, dim=0)

    # --------------------------------------------------------------- forward
    def forward(self, tokenized, length, max_len, init_hidden):
        pad_history = self._pad_history_buf
        self._pad_history_buf = None  # consume

        if self.cache is not None and pad_history is not None:
            # Cached path: skip the BERT forward and assemble (T, B, 768)
            # CLS reps directly from the cache (filling misses on demand).
            sentence_rep = self._encode_cls_cached(pad_history)  # (B*T, 768)
            sentence_rep = sentence_rep.view(-1, self.num_turns, 768).permute(1, 0, 2)
        else:
            bert_in = {
                k: v
                for k, v in tokenized.items()
                if isinstance(v, torch.Tensor)
            }
            bert_embed = (
                self.model(**bert_in)[0]
                .view(-1, self.num_turns, max_len, 768)
                .permute(1, 0, 2, 3)
            )  # (num_turns, batch_size, max_len, 768)
            sentence_rep = bert_embed[:, :, 0, :]

        output, hidden = self.rnn(sentence_rep, init_hidden)
        output = output[-1, :, :]
        return output

    # ----------------------------------------------------------- prepare_data
    def prepare_data(self, dialog_history, device, raw_history=False):
        def normalize_turn(turn):
            if isinstance(turn, str):
                return turn
            if isinstance(turn, (list, tuple)):
                return " ".join(str(tok) for tok in turn)
            if turn is None:
                return ""
            return str(turn)

        pad_history = []
        word_index = []
        word_batch_index = []
        all_words = []
        for i in range(len(dialog_history)):
            turn_words = []
            cur_history = ""
            normalized_history = [normalize_turn(turn) for turn in dialog_history[i]]
            turns = len(normalized_history)
            if turns > self.num_turns:
                padded = normalized_history[turns - self.num_turns :]
                pad_history = pad_history + padded
                for sen in padded:
                    cur_history = cur_history + sen + " "
            elif turns < self.num_turns:
                pad = []
                for j in range(self.num_turns - turns):
                    pad.append("")
                padded = pad + normalized_history
                for sen in normalized_history:
                    cur_history = cur_history + sen + " "

                pad_history = pad_history + padded
            else:
                pad_history = pad_history + normalized_history
                for sen in normalized_history:
                    cur_history = cur_history + sen + " "

            token_history = word_tokenize(cur_history)
            if len(token_history) > self.num_words:
                token_history = token_history[-self.num_words :]
            elif len(token_history) == 0:
                word_index.append(0)
                word_batch_index.append(i)

            for word in token_history:
                word_index.append(self.key2index.get(word.lower(), 0))
                word_batch_index.append(i)
                turn_words.append(self.key2index.get(word.lower(), 0))
            all_words.append(turn_words)

        tokenized_dialog = self.tokenizer(
            pad_history,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )  # [turn1,turn2,turn3....]
        for key in tokenized_dialog.keys():
            tokenized_dialog[key] = tokenized_dialog[key].to(device=device)
        all_length = (
            torch.sum(tokenized_dialog["attention_mask"], dim=-1)
            .view(-1, self.num_turns)
            .permute(1, 0)
        )
        maxlen = tokenized_dialog["attention_mask"].size()[-1]
        batch_size = len(dialog_history)

        if self.rnn_type == "LSTM":
            init_hidden = (
                torch.zeros(1, batch_size, self.rnn_hidden),
                torch.zeros(batch_size, 1, self.rnn_hidden),
            ).to(device=device)
        else:
            init_hidden = torch.zeros(1, batch_size, self.rnn_hidden).to(device=device)

        # Stash raw text list so the cached forward path can look them up.
        # `forward` consumes and clears this immediately after the call.
        self._pad_history_buf = pad_history

        w_index = None
        w_batch_index = None

        if self.word_net:
            w_index = torch.Tensor(word_index).long().to(device=device)
            w_batch_index = torch.Tensor(word_batch_index).long().to(device=device)
        if raw_history:
            return (
                tokenized_dialog,
                all_length,
                maxlen,
                init_hidden,
                w_index,
                w_batch_index,
                all_words,
            )
        else:
            return (
                tokenized_dialog,
                all_length,
                maxlen,
                init_hidden,
                w_index,
                w_batch_index,
            )
