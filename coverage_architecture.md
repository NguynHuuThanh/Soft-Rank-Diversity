# Coverage Implementation and Loss Architecture in CR-Walker

This note describes how coverage is computed in the current implementation, and where the diversity loss is injected into the model architecture.

## 1) Coverage metric implementation

Coverage is computed inside `model/evaluation.py`.

### ReDial coverage

For ReDial, the model scores the full movie catalog. Coverage is measured by collecting all unique items that appear in the top-k recommendations across the test set.

```python
def compute_item_coverage_redial(all_scores_list, n_movies):
    """
    Item Coverage@10 and Item Coverage@50 for ReDial.
    Coverage@k = fraction of catalog items that appear at least once in
    any sample's top-k recommendation list.
    """
    all_top10_items = set()
    all_top50_items = set()

    for scores in all_scores_list:
        if len(scores) != n_movies:
            continue
        top50_idx = np.argsort(scores)[-50:]
        top10_idx = top50_idx[-10:]
        all_top10_items.update(top10_idx.tolist())
        all_top50_items.update(top50_idx.tolist())

    if n_movies > 0:
        coverage_10 = len(all_top10_items) / n_movies
        coverage_50 = len(all_top50_items) / n_movies
    else:
        coverage_10 = 0
        coverage_50 = 0

    return {
        'item_coverage@10': coverage_10,
        'item_coverage@50': coverage_50
    }
```

### GoRecDial coverage

For GoRecDial, each recommendation turn uses a fixed candidate set. Coverage is computed over the unique candidate items that appear in the output lists.

```python
def compute_item_coverage_gorecdial(all_rec_lists, n_movies):
    """
    Item Coverage for GoRecDial.
    Coverage@k = fraction of catalog items that appear at least once in
    any sample's top-k recommendation list.
    """
    all_top_items = set()
    for reclist in all_rec_lists:
        all_top_items.update(reclist)

    if n_movies > 0:
        coverage = len(all_top_items) / n_movies
    else:
        coverage = 0

    return {
        'item_coverage': coverage
    }
```

### KG coverage helper

The current evaluation also computes KG-entity coverage from the item neighborhoods in the graph:

```python
def compute_kg_coverage(topk_item_ids, movie_kg_neighbors, total_reachable_entities):
    """
    KG-Entity Coverage@k: fraction of reachable KG entities covered by
    the 1-hop neighborhoods of the top-k items.
    """
    covered = set()
    for item_id in topk_item_ids:
        covered |= movie_kg_neighbors.get(item_id, set())
    return len(covered) / total_reachable_entities
```

The evaluation loop accumulates these values per turn and per dialog, then averages them at the end.

---

## 2) Where the loss is added in the architecture

The main architecture lives in `model/CR_walker.py` inside the `ProRec` class.

The base training objective is assembled in:

- `ProRec.forward(...)` for ReDial
- `ProRec.forward_gorecdial(...)` for GoRecDial

The coverage-related diversity loss is **not** the main recommendation loss. It is added **after** the core losses are computed and **before** the final return.

### ReDial path

In `forward(...)`, the model computes:

- `walk_loss_1`
- `walk_loss_2`
- `intent_loss`
- alignment regularization loss
- optional diversity loss
- optional DPP loss

The diversity loss is inserted here:

```python
tot_loss = walk_loss_1 + walk_loss_2 + intent_loss + 0.025 * reg_loss

# --- Diversity loss for ReDial ---
if self.div_loss_weight > 0:
    batch_size = utter_embed.size(0)
    l2_scores = paths[1]
    l2_sel = sel_indices[1]
    l2_bat = sel_batch_indices[1]
    l2_mask = score_masks[1]

    mc = graph_embed.size(0)
    dense_scores = torch.full((batch_size, mc), -1e9, device=self.device)
    dense_mask = torch.zeros((batch_size, mc), device=self.device)
    dense_scores[l2_bat, l2_sel] = torch.max(dense_scores[l2_bat, l2_sel], l2_scores)
    dense_mask[l2_bat, l2_sel] = l2_mask

    has_rec = dense_mask.sum(dim=-1) > 0
    if has_rec.any():
        rec_scores = dense_scores[has_rec]
        rec_mask = dense_mask[has_rec]
        div_loss = self.compute_diversity_loss(rec_scores, graph_embed, mask=rec_mask)
        tot_loss = tot_loss + self.div_loss_weight * div_loss
```

The DPP loss is also added later in the same function:

```python
if self.dpp_loss_weight > 0:
    ...
    tot_loss = tot_loss + self.dpp_loss_weight * dpp_loss
```

### GoRecDial path

In `forward_gorecdial(...)`, the model computes:

- `walk_loss_1`
- `walk_loss_2`
- `intent_loss`
- `rec_loss`
- alignment regularization loss
- optional diversity loss

The diversity loss is injected here:

```python
tot_loss = walk_loss_1 + walk_loss_2 + intent_loss + rec_loss + 0.025 * reg_loss

# --- Diversity loss for GoRecDial ---
if self.div_loss_weight > 0:
    batch_size = rec.size(0)
    cand_ids = rec_index.view(batch_size, 5)
    cand_embeds = graph_embed[cand_ids]
    div_loss_sum = torch.tensor(0.0, device=self.device)
    for b in range(batch_size):
        div_loss_sum = div_loss_sum + self.compute_diversity_loss(
            rec[b:b+1],
            cand_embeds[b]
        )
    div_loss = div_loss_sum / batch_size
    tot_loss = tot_loss + self.div_loss_weight * div_loss
```

So the architecture is:

1. Encode the dialog with the utterance encoder.
2. Build graph embeddings with the graph embedder.
3. Predict intent.
4. Perform graph walking / recommendation.
5. Compute the base losses.
6. Add the diversity loss if `div_loss_weight > 0`.
7. Optionally add DPP loss for ReDial if `dpp_loss_weight > 0`.
8. Return the final `tot_loss`.

---

## 3) How to enable the coverage-aware training

The diversity loss is controlled by command-line arguments in `model/train_redial.py`:

- `--div_loss_weight`
- `--dpp_loss_weight`
- `--div_temperature`

Example:

```bash
python .\model\train_redial.py --option train --model_name redial_reason_128 --div_loss_weight 1.23
```

If `--div_loss_weight 0`, the diversity loss is disabled.

---

## 4) Practical interpretation

- **Recall** measures recommendation correctness.
- **Item coverage** measures how many different items appear across top-k predictions.
- **KG coverage** measures how much of the graph neighborhood is exposed by the recommendations.
- The diversity loss is designed to increase coverage, but it may slightly reduce recall.

That trade-off is expected and matches the behavior seen in the training logs.
