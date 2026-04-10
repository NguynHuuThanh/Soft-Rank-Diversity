"""
TG-ReDial PyG dataset for the CR-Walker pipeline.

Mirrors the layout of `data/redial.py` so the rest of the training code can
treat it as a drop-in alternative once the model side is wired up via
`model/conf.py` (`dataset='tgredial'`) and `model/train_tgredial.py`.

Layout assumed under root/data/tgredial:
    raw/tgredial_train.json
    raw/tgredial_test.json
    raw/tgredial_kg.json
    raw/movie_ids.json

NOTE — known differences vs ReDial:

* TG-ReDial is Chinese; `dialog_history` carries Chinese tokens. The
  English-only BERT in `utterance_embedder.py` will need to be swapped for
  a Chinese checkpoint (e.g. `bert-base-chinese`) before training succeeds.
* The KG ships as `{entity, edge, n_relation}` (54,788 entities,
  35 relation types) with no per-node "type" tag. We synthesise a minimal
  2-dim node feature: [is_movie, is_other], because CR-Walker only consumes
  this through `node_feature` and downstream code reads it as a generic
  embedding.
* Intent labels are not part of the raw JSON in the same form as ReDial;
  they are produced by `model.intent_classifier_zh.ChineseIntentClassifier`,
  preferring the dataset's own policy hint when present.
* `node_candidate1/2`, `label_1/2`, `gold_pos`, `label_rec` are derived
  from `target`, `context_entities`, `ent_rec`, and `movie_rec`.
"""

import os.path as osp
import json
import sys
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm

# Make `model/` importable for the intent classifier helper.
_PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from model.intent_classifier_zh import ChineseIntentClassifier
except Exception:  # pragma: no cover - keep import-time failures soft
    ChineseIntentClassifier = None  # type: ignore


def _load_processed_data(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def _flatten_tokens(context_tokens):
    """ReDial stores context as List[List[str]]; we keep the same shape."""
    if not context_tokens:
        return []
    if isinstance(context_tokens[0], list):
        return context_tokens
    return [context_tokens]


def _diff_entities(curr, prev):
    prev_set = set(prev)
    return [e for e in curr if e not in prev_set]


def build_id_remap(num_entities, movie_ids):
    """Return ``old_to_new`` mapping such that movies occupy positions
    ``0..len(movies)-1`` and non-movies occupy ``len(movies)..N-1``.

    The CR-Walker pipeline assumes "movies" are the first contiguous block
    of entity ids (true for ReDial, false for TG-ReDial). Remapping fixes
    eval (`movie_count` == real movie count) and keeps `label_rec` ids
    aligned with score-vector positions.

    The mapping is deterministic (sorted by original id) so the data
    loader and `conf.preprocess_tgredial` produce identical results when
    they each compute it independently.
    """
    movie_set = set(int(m) for m in movie_ids)
    sorted_movies = sorted(m for m in movie_set if 0 <= m < num_entities)
    sorted_others = sorted(i for i in range(num_entities) if i not in movie_set)
    old_to_new = [-1] * num_entities
    for new_id, old in enumerate(sorted_movies):
        old_to_new[old] = new_id
    offset = len(sorted_movies)
    for new_id, old in enumerate(sorted_others):
        old_to_new[old] = offset + new_id
    return old_to_new, len(sorted_movies)


class TGReDial(InMemoryDataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        flag="train",
        intent_classifier: "ChineseIntentClassifier | None" = None,
    ):
        self._intent_clf = intent_classifier
        super().__init__(root, transform, pre_transform)
        self.flag = flag
        if self.flag == "test":
            self.data, self.slices = _load_processed_data(self.processed_paths[1])
        elif self.flag == "graph":
            self.data, self.slices = _load_processed_data(self.processed_paths[2])
        elif self.flag == "rec":
            self.data, self.slices = _load_processed_data(self.processed_paths[3])
        else:
            self.data, self.slices = _load_processed_data(self.processed_paths[0])

        self._ensure_num_nodes_attr()

    # ---- mirror ReDial helper for older PyG versions ----
    def _ensure_num_nodes_attr(self):
        if self.flag == "graph":
            return
        if hasattr(self.data, "num_nodes"):
            return
        if not hasattr(self, "slices") or self.slices is None:
            return
        if "my_id" not in self.slices:
            return
        num_examples = int(self.slices["my_id"].numel() - 1)
        if num_examples <= 0:
            return
        self.data.num_nodes = torch.ones(num_examples, dtype=torch.long)
        self.slices["num_nodes"] = torch.arange(num_examples + 1, dtype=torch.long)

    @property
    def raw_dir(self):
        # The TG-ReDial JSONs ship directly under data/tgredial/ (no raw/
        # subfolder). Point PyG at the root so `raw_paths` resolves.
        return self.root

    @property
    def raw_file_names(self):
        # tgredial_kg.json lives one directory up (data/tgredial_kg.json),
        # not inside data/tgredial/. PyG only checks `raw_file_names` for
        # existence, so we list the files that *do* live in raw_dir here
        # and resolve the KG path explicitly inside process().
        return [
            "tgredial_test.json",
            "tgredial_train.json",
            "movie_ids.json",
        ]

    @property
    def processed_file_names(self):
        return ["train.pt", "test.pt", "graph.pt", "rec.pt"]

    def download(self):
        return

    # ------------------------------------------------------------------ KG
    def _build_graph(self, kg, old_to_new, n_movies):
        entities = kg["entity"]
        edges = kg["edge"]
        n_relation = int(kg.get("n_relation", 0))
        num_nodes = len(entities)

        # node_feature row order follows the *new* (post-remap) ids:
        #   positions 0..n_movies-1 -> Movie
        #   positions n_movies..N-1  -> Other
        node_feature = torch.zeros(num_nodes, 2)
        node_feature[:n_movies, 0] = 1.0
        node_feature[n_movies:, 1] = 1.0

        edge_index = [[], []]
        edge_type = []
        for src, dst, rel in edges:
            s, d = int(src), int(dst)
            if s >= num_nodes or d >= num_nodes:
                continue  # skip the 12 stray edges referencing unknown ids
            edge_index[0].append(old_to_new[s])
            edge_index[1].append(old_to_new[d])
            edge_type.append(int(rel))

        edge_index = torch.from_numpy(np.array(edge_index)).long()
        edge_type = torch.from_numpy(np.array(edge_type)).long()

        return Data(
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=num_nodes,
            graph_size=num_nodes,
            node_feature=node_feature,
            num_movies=n_movies,
            n_relation=n_relation,
        )

    # ----------------------------------------------------------- per-record
    def _remap_ids(self, ids):
        out = []
        for e in ids:
            try:
                ei = int(e)
            except (TypeError, ValueError):
                continue
            if 0 <= ei < len(self._old_to_new):
                v = self._old_to_new[ei]
                if v >= 0:
                    out.append(v)
        return out

    def _build_record(self, idx, rec, prev_entities, last_turn, intent_clf):
        context_tokens = _flatten_tokens(rec.get("context_tokens") or [])
        context_entities = self._remap_ids(rec.get("context_entities") or [])
        new_mention = _diff_entities(context_entities, prev_entities)

        # Pull a native intent hint from `target` (Recommender turns) or
        # the latest `context_policy` step (Seeker turns) if any.
        native_hint = None
        target = rec.get("target") or []
        if target and isinstance(target[0], (list, tuple)) and target[0]:
            native_hint = target[0][0]
        else:
            policy = rec.get("context_policy") or []
            for step in reversed(policy):
                if step and isinstance(step[0], (list, tuple)) and step[0]:
                    native_hint = step[0][0]
                    break

        flat_text = "".join(["".join(toks) for toks in context_tokens])
        intent = (
            intent_clf.predict(flat_text, native_hint=native_hint)
            if intent_clf is not None
            else (ChineseIntentClassifier.from_native_policy(native_hint) or "chat")
        )

        # Candidate / label fields. TG-ReDial doesn't ship CR-Walker style
        # 2-step candidate lists, so we approximate:
        #   node_candidate1 = entities mentioned anywhere in the context
        #                     (these are what the walker selects from)
        #   node_candidate2 = same set (placeholder for the depth-2 step)
        #   label_1/2       = positions of gold target entities within
        #                     node_candidate1, if present
        target_entities = []
        if target and isinstance(target[0], (list, tuple)) and len(target[0]) > 1:
            target_entities = self._remap_ids(target[0][1] or [])

        node_candidate1 = list(dict.fromkeys(context_entities + target_entities)) or [0]

        cand_index = {e: i for i, e in enumerate(node_candidate1)}
        label_1 = [cand_index[e] for e in target_entities if e in cand_index]

        # graph_walker expects node_candidate2 / label_2 as list-of-lists
        # (one depth-2 group per depth-1 pick). TG-ReDial has no depth-2
        # supervision, so emit empty groups — graph_walker has a `len(item)==0`
        # branch that handles this by padding with null_idx.
        node_candidate2 = [[] for _ in label_1]
        label_2 = [[] for _ in label_1]

        movie_rec = self._remap_ids(rec.get("movie_rec") or [])
        items = self._remap_ids(rec.get("items") or [])
        is_recommend = intent == "recommend" and bool(items or movie_rec)
        gold_pos = items if is_recommend else []
        label_rec = items if is_recommend else []

        key = f"{rec.get('user_id', 0)}_{rec.get('time_id', idx)}_{idx}"

        return Data(
            dialog_history=context_tokens,
            oracle_response=rec.get("response", []),
            mention_history=context_entities,
            node_candidate1=node_candidate1,
            label_1=label_1,
            node_candidate2=node_candidate2,
            label_2=label_2,
            intent=intent,
            new_mention=new_mention,
            my_id=key,
            last_turn=last_turn,
            gold_pos=gold_pos,
            label_rec=label_rec,
            num_nodes=1,
        ), is_recommend

    # ------------------------------------------------------------- process
    def process(self):
        test_path, train_path, movie_path = self.raw_paths
        # KG file lives one level above the dataset root.
        graph_path = osp.join(osp.dirname(self.root), "tgredial_kg.json")

        with open(test_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        with open(train_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        with open(graph_path, "r", encoding="utf-8") as f:
            kg = json.load(f)
        with open(movie_path, "r", encoding="utf-8") as f:
            movie_ids = json.load(f)

        # Build the entity-id remap once and stash on self so _build_record
        # / _build_graph can both consume it.
        self._old_to_new, self._n_movies = build_id_remap(len(kg["entity"]), movie_ids)

        if self._intent_clf is None and ChineseIntentClassifier is not None:
            cache_dir = osp.join(self.root, "processed")
            self._intent_clf = ChineseIntentClassifier(
                cache_path=osp.join(cache_dir, "intent_cache.json"),
                use_model=True,
            )

        flag_to_records = [("train", train_data), ("test", test_data)]
        rec_list = []

        for q, (flag, records) in enumerate(flag_to_records):
            data_list = []
            prev_user = None
            prev_entities: list = []
            for idx in tqdm(range(len(records)), desc=f"tgredial:{flag}"):
                rec = records[idx]
                user_id = rec.get("user_id")
                if user_id != prev_user:
                    prev_entities = []
                    prev_user = user_id

                # CR-Walker-style training is only meaningful for the
                # system side (Recommender). Skip Seeker turns to match
                # how ReDial frames the task.
                if rec.get("role") != "Recommender":
                    prev_entities = list(rec.get("context_entities") or prev_entities)
                    continue

                last_turn = 0
                if idx == len(records) - 1:
                    last_turn = 1
                else:
                    nxt = records[idx + 1]
                    if nxt.get("user_id") != user_id:
                        last_turn = 1

                data, is_recommend = self._build_record(
                    idx, rec, prev_entities, last_turn, self._intent_clf
                )

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
                if flag == "test" and is_recommend:
                    rec_list.append(data)

                prev_entities = list(rec.get("context_entities") or prev_entities)

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[q])

        rec_data, rec_slices = self.collate(rec_list)
        torch.save((rec_data, rec_slices), self.processed_paths[3])

        graph_data = self._build_graph(kg, self._old_to_new, self._n_movies)
        graph_pack, graph_slices = self.collate([graph_data])
        torch.save((graph_pack, graph_slices), self.processed_paths[2])

        if self._intent_clf is not None:
            self._intent_clf.save_cache()


if __name__ == "__main__":
    root = osp.dirname(osp.dirname(osp.abspath(__file__)))
    path = osp.join(root, "data", "tgredial")
    ds = TGReDial(path, flag="train")
    print("train size:", len(ds))
    sample = ds[0]
    print("intent:", sample.intent)
    print("nc1:", sample.node_candidate1[:10])
    print("label_1:", sample.label_1)
    print("gold_pos:", sample.gold_pos)
