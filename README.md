# CR-Walker

Code for paper "CR-Walker: Conversational Recommender System with Tree-structured Graph Reasoning and Dialog Acts" EMNLP 2021.

you can find our paper at [arxiv](https://arxiv.org/abs/2010.10333).

Cite this paper:

```
@inproceedings{ma2021crwalker,
  title={CR-Walker: Tree-Structured Graph Reasoning and Dialog Acts for Conversational Recommendation},
  author={Ma, Wenchang and Takanobu, Ryuichi and Huang, Minlie},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages={1839--1851},
  year={2021},
  organization={ACL}
}
```



## Data

- [google link](https://drive.google.com/drive/folders/1Jg65ibsj_2tybZyCQnGD7y9a80FlCX61?usp=sharing) to raw data and our model checkpoints. Table of content: 

  ```
  CR-Walker
  ├─data
  │  ├─gorecdial
  │  │  └─raw
  │  ├─gorecdial_gpt
  │  ├─redial
  │  │  └─raw
  │  └─redial_gpt
  └─saved
  ```

- download to [your home directory]/CR-Walker/.

## Train

- **For GoRecdial**: 

  ```
  python train_gorecdial.py --option train --model_name <your_model_name> --pretrain
  ```

- **For Redial**: 

  ```
  python train_redial.py --option train --model_name <your_model_name> --pretrain 
  ```

  We implemented an MIM pretraining stage similar to [KGSF](https://arxiv.org/abs/2007.04032) to accelerate training. Also, we provided option of adding wordnet features by adding "--word_net" as command line option.


## Test Recommendation

- **For GoRecdial**

  ```
  python train_gorecdial.py --option test --model_name gorecdial_reason_128
  ```

- **For Redial**:  

  ```
  python train_redial.py --option test --model_name redial_reason_128
  ```

  You can directly evaluate the best model checkpoints for the two datasets that we provided. The results may slightly differ from the paper since we re-trained the model. Note that the reasoning width (*'sample'* argument in **conf.py**) has been set to 1 for speed during training. You can tune it larger along with the selection threshold (*'threshold'* argument in **conf.py**) to yield better performance.


## Test Generation

- **For GoRecdial**

  ```
  python train_gorecdial.py --option test_gen --model_name gorecdial_reason_128
  ```

- **For Redial**:  

  ```
  python train_redial.py --option test_gen --model_name redial_reason_128
  ```

  Similarly, you can tune the selection threshold, reasoning width and max number of leaf nodes (*'max_leaf'* argument in **conf.py**) to control generation. 


## Requirements

python==3.6.10

pytorch==1.4.0

torch_geometric==1.6.0

```
for w in 0.2 0.5 0.7 1.0 1.5 2.0; do
  BERT_MODEL_NAME=bert-base-chinese python train_tgredial.py --option train \
    --model_name tg_softild_${w//./p} \
    --div_loss_weight $w --div_temperature 0.1
done
```

## Save-model and metric logic (single-file reference)

This project computes `f1@k` and `item_coverage@k` in `model/evaluation.py` and uses those values to save best checkpoints inside `model/train_tgredial.py` (same pattern in `model/train_redial.py`).

### 1) How `f1@1/10/50` is computed

In `evaluate_rec_redial(...)`:

- Recall:
  - `recall@k = hit_k / tot_rec`
- Precision:
  - `precision@1 = hit_1 / (recommend_turns * 1)`
  - `precision@10 = hit_10 / (recommend_turns * 10)`
  - `precision@50 = hit_50 / (recommend_turns * 50)`
- F1 per k:
  - `f1@k = 0` when `precision@k + recall@k == 0`
  - otherwise `f1@k = 2 * precision@k * recall@k / (precision@k + recall@k)`

Returned as:

```python
f1_results = {
    'f1@1': f1_1,
    'f1@10': f1_10,
    'f1@50': f1_50,
}
```

### 2) How `item_coverage@1/10/50` is computed

In `compute_item_coverage_redial(all_scores_list, n_movies)`:

1. For each recommendation score vector (`scores`) with length `n_movies`:
   - `top50_idx = np.argsort(scores)[-50:]`
   - `top10_idx = top50_idx[-10:]`
   - `top1_idx = top50_idx[-1:]`
2. Add those indices into global sets:
   - `all_top1_items`, `all_top10_items`, `all_top50_items`
3. Coverage is unique recommended items over catalog size:
   - `item_coverage@1 = len(all_top1_items) / n_movies`
   - `item_coverage@10 = len(all_top10_items) / n_movies`
   - `item_coverage@50 = len(all_top50_items) / n_movies`

Then `evaluate_rec_redial(...)` merges it into `coverage_results` and prints:

```python
item_coverage_results = compute_item_coverage_redial(all_scores_list, args['movie_count'])
coverage_results.update(item_coverage_results)
```

### 3) When model checkpoints are saved

In training (`train_tgredial.py` / `train_redial.py`), best-so-far values are tracked:

- Recall checkpoints:
  - `best_recall_1`, `best_recall_10`, `best_recall_50`
- F1 checkpoints:
  - `best_f1_1`, `best_f1_10`, `best_f1_50`
- Item-coverage checkpoints:
  - `best_coverage_1`, `best_coverage_10`, `best_coverage_50`

On every evaluation pass (both mid-epoch eval and epoch-end eval), if current metric is strictly greater than historical best, save current state dict:

```python
if f1_results['f1@50'] > best_f1_50:
    best_f1_50 = f1_results['f1@50']
    torch.save(prorec.state_dict(), save_path_f1_50)

if coverage_results['item_coverage@10'] > best_coverage_10:
    best_coverage_10 = coverage_results['item_coverage@10']
    torch.save(prorec.state_dict(), save_path_cov10)
```

Checkpoint filenames:

- Overall/baseline: `saved/best_model_<model_name>.pt`
- Recall-specific: `..._1.pt`, `..._10.pt`, `..._50.pt`
- Coverage-specific: `..._cov1.pt`, `..._cov10.pt`, `..._cov50.pt`
- F1-specific: `..._f1at1.pt`, `..._f1at10.pt`, `..._f1at50.pt`

This is exactly why logs such as `f1@50 new high ... saving model...` and `item_coverage@10 new high ... saving model...` appear during training.