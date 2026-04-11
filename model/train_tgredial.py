"""
TG-ReDial training entry point — paralleled from `train_redial.py`.

Differences vs train_redial.py:
  * Loads the TGReDial PyG dataset (Chinese, alternate KG schema).
  * Calls `add_generic_args('tgredial')` so paths/none_node/preprocess
    are wired through the tgredial branch in conf.py.
  * Intent labels for each turn are produced by a pretrained Chinese
    zero-shot classifier (model.intent_classifier_zh) at preprocess time;
    no CR-Walker-style labels are required.

Known caveats — *not* fixed in this script, the user will need to address
them before training converges:
  - `utterance_embedder.py` hardcodes `bert-base-uncased`; switch to
    `bert-base-chinese` (or another Chinese checkpoint) when running on
    TG-ReDial.
  - `CR_walker.prepare_data_redial` assumes ReDial-style fields; the
    TGReDial loader produces the same field names so the call below
    should work, but `attribute_dict`-driven candidate sampling will
    behave differently because TG-ReDial has ~54k entities vs ~30k.
"""

import sys
import os.path as osp
import argparse
import json
import time
from termcolor import colored

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch_geometric.loader import DataLoader

from CR_walker import ProRec
from evaluation import evaluate_rec_redial, evaluate_gen_redial
from conf import add_generic_args, args
from data.tgredial import TGReDial


device_str = 'cuda:0'
device = torch.device(device_str)


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default='tgredial_reason_1best', type=str,
                    help="model name for saving stats and parameters")
parser.add_argument("--option", choices=['train', 'test', 'test_gen'], default='train')
parser.add_argument("--pretrain", action='store_true')
parser.add_argument("--restore_best", action='store_true')
parser.add_argument("--graph_embed_size", type=int, default=128)
parser.add_argument("--utter_embed_size", type=int, default=128)
parser.add_argument("--negative_sample_ratio", type=int, default=5)
parser.add_argument("--train_epoch", type=int, default=60)
parser.add_argument("--pretrain_epoch", type=int, default=3)
parser.add_argument("--atten_hidden", type=int, default=20)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--eval_batch", type=int, default=5000)
parser.add_argument("--word_net", action='store_true')
parser.add_argument("--div_loss_weight", type=float, default=0.0)
parser.add_argument("--dpp_loss_weight", type=float, default=0.0)
parser.add_argument("--div_temperature", type=float, default=1)
parser.add_argument("--coverage_topk", type=str, default="1,10,50")
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("--early_stop_patience", type=int, default=5)
t_args = parser.parse_args()


option = t_args.option
model_name = t_args.model_name

root = osp.dirname(osp.dirname(osp.abspath(__file__)))
save_path = osp.join(root, "saved", "best_model_" + model_name + ".pt")
save_path_1 = osp.join(root, "saved", "best_model_" + model_name + "_1.pt")
save_path_10 = osp.join(root, "saved", "best_model_" + model_name + "_10.pt")
save_path_50 = osp.join(root, "saved", "best_model_" + model_name + "_50.pt")
save_path_cov1 = osp.join(root, "saved", "best_model_" + model_name + "_cov1.pt")
save_path_cov10 = osp.join(root, "saved", "best_model_" + model_name + "_cov10.pt")
save_path_cov50 = osp.join(root, "saved", "best_model_" + model_name + "_cov50.pt")
save_path_f1_1 = osp.join(root, "saved", "best_model_" + model_name + "_f1at1.pt")
save_path_f1_10 = osp.join(root, "saved", "best_model_" + model_name + "_f1at10.pt")
save_path_f1_50 = osp.join(root, "saved", "best_model_" + model_name + "_f1at50.pt")

path = osp.join(root, "data", "tgredial")
tg_train = TGReDial(path, flag="train")
tg_test = TGReDial(path, flag="test")
tg_graph = TGReDial(path, flag="graph")
tg_rec = TGReDial(path, flag="rec")
graph_data = tg_graph[0]


train_loader = DataLoader(tg_train, batch_size=20, shuffle=True)
test_loader = DataLoader(tg_test, batch_size=20, shuffle=False)

add_generic_args('tgredial')
args['div_loss_weight'] = t_args.div_loss_weight
args['dpp_loss_weight'] = t_args.dpp_loss_weight
args['div_temperature'] = t_args.div_temperature
args['coverage_topk'] = [int(x) for x in t_args.coverage_topk.split(',')]


if option == "train":
    prorec = ProRec(
        device_str=device_str,
        dataset="tgredial",
        graph_embed_size=t_args.graph_embed_size,
        utter_embed_size=t_args.utter_embed_size,
        negative_sample_ratio=t_args.negative_sample_ratio,
        atten_hidden=t_args.atten_hidden,
        word_net=t_args.word_net,
        div_loss_weight=t_args.div_loss_weight,
        div_temperature=t_args.div_temperature,
        dpp_loss_weight=t_args.dpp_loss_weight,
    )
    best_coverage_1 = 0
    best_coverage_10 = 0
    best_coverage_50 = 0
    best_f1_1 = 0
    best_f1_10 = 0
    best_f1_50 = 0
    if t_args.restore_best:
        print("restoring from best checkpoint...")
        state_dict = torch.load(save_path)
        prorec.load_state_dict(state_dict, strict=False)
        with open('stats_' + model_name + '.json') as f:
            stats_all = json.load(f)
        best_recall_1 = max(stats_all.get('recall_1', [0]) or [0])
        best_recall_10 = max(stats_all.get('recall_10', [0]) or [0])
        best_recall_50 = max(stats_all.get('recall_50', [0]) or [0])
        for _k in args['coverage_topk']:
            for _m in ['kg_cov_turn', 'kg_cov_dialog']:
                stats_all.setdefault(f'{_m}@{_k}', [])
        stats_all.setdefault('item_coverage@1', [])
        stats_all.setdefault('item_coverage@10', [])
        stats_all.setdefault('item_coverage@50', [])
        stats_all.setdefault('f1@1', [])
        stats_all.setdefault('f1@10', [])
        stats_all.setdefault('f1@50', [])

        if len(stats_all['item_coverage@1']) > 0:
            best_coverage_1 = max(stats_all['item_coverage@1'])
        if len(stats_all['item_coverage@10']) > 0:
            best_coverage_10 = max(stats_all['item_coverage@10'])
        if len(stats_all['item_coverage@50']) > 0:
            best_coverage_50 = max(stats_all['item_coverage@50'])
        if len(stats_all['f1@1']) > 0:
            best_f1_1 = max(stats_all['f1@1'])
        if len(stats_all['f1@10']) > 0:
            best_f1_10 = max(stats_all['f1@10'])
        if len(stats_all['f1@50']) > 0:
            best_f1_50 = max(stats_all['f1@50'])
    else:
        best_recall_1 = 0
        best_recall_10 = 0
        best_recall_50 = 0
        stats_all = {"recall_1": [], "recall_10": [], "recall_50": []}
        for _k in args['coverage_topk']:
            for _m in ['kg_cov_turn', 'kg_cov_dialog']:
                stats_all[f'{_m}@{_k}'] = []
        stats_all['item_coverage@1'] = []
        stats_all['item_coverage@10'] = []
        stats_all['item_coverage@50'] = []
        stats_all['f1@1'] = []
        stats_all['f1@10'] = []
        stats_all['f1@50'] = []

    unfreeze_layers = ["utter_embedder.rnn", "intent_selector", "graph_embedder",
                       "graph_walker", "Wa", "Ww"]
    for name, param in prorec.named_parameters():
        param.requires_grad = False
        for ele in unfreeze_layers:
            if ele in name:
                param.requires_grad = True
                print(name)
                break

    optimizer = torch.optim.Adam(prorec.parameters(), lr=t_args.lr,
                                 weight_decay=t_args.weight_decay)
    prorec.to(device)

    num = 0
    num_pretrain = 0
    pretrain_epoch = t_args.pretrain_epoch
    max_epoch = t_args.train_epoch

    best_epoch_recall10 = -1.0
    no_improve_epochs = 0
    prev_epoch_avg_loss = None

    if t_args.pretrain:
        for i in range(pretrain_epoch):
            for batch in train_loader:
                optimizer.zero_grad()
                (tokenized_dialog, all_length, maxlen, init_hidden, edge_type,
                 edge_index, alignment_index, alignment_batch_index,
                 alignment_label, intent_label, alignment_index_word,
                 alignment_batch_index_word, alignment_label_word) = prorec.prepare_pretrain(
                    batch.new_mention, batch.dialog_history, batch.intent,
                    graph_data.edge_type, graph_data.edge_index,
                )
                loss = prorec.forward_pretrain(
                    tokenized_dialog, all_length, maxlen, init_hidden, edge_type,
                    edge_index, alignment_index, alignment_batch_index,
                    alignment_label, intent_label, alignment_index_word,
                    alignment_batch_index_word, alignment_label_word,
                )
                loss.backward()
                optimizer.step()
                print("pretrain iter ", num_pretrain, ":", loss.item())
                num_pretrain += 1

    for i in range(max_epoch):
        epoch_start_time = time.time()
        prorec.train()
        epoch_loss_sum = 0.0
        epoch_steps = 0
        epoch_base_loss_sum = 0.0
        epoch_div_loss_sum = 0.0
        epoch_dpp_loss_sum = 0.0
        epoch_final_loss_sum = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            (tokenized_dialog, all_length, maxlen, init_hidden, edge_type,
             edge_index, mention_index, mention_batch_index, sel_indices,
             sel_batch_indices, sel_group_indices, grp_batch_indices,
             last_indices, intent_indices, intent_label, label_1, label_2,
             score_masks, word_index, word_batch_index) = prorec.prepare_data_redial(
                batch.dialog_history, batch.mention_history, batch.intent,
                batch.node_candidate1, batch.node_candidate2,
                graph_data.edge_type, graph_data.edge_index,
                batch.label_1, batch.label_2, batch.gold_pos,
                args['attribute_dict'], sample=True,
            )
            (alignment_index, alignment_batch_index, alignment_label,
             alignment_index_word, alignment_batch_index_word,
             alignment_label_word) = prorec.prepare_reg(
                batch.new_mention, batch.dialog_history, batch.intent,
            )

            intent, paths, loss = prorec.forward(
                tokenized_dialog, all_length, maxlen, init_hidden, edge_type,
                edge_index, mention_index, mention_batch_index, sel_indices,
                sel_batch_indices, sel_group_indices, grp_batch_indices,
                last_indices, intent_indices, intent_label, label_1, label_2,
                score_masks, alignment_index, alignment_batch_index,
                alignment_label, word_index, word_batch_index,
                alignment_index_word, alignment_batch_index_word,
                alignment_label_word,
            )

            loss.backward()
            optimizer.step()

            loss_val = float(loss.item())
            epoch_loss_sum += loss_val
            epoch_steps += 1

            loss_terms = getattr(prorec, 'last_train_loss_terms', None)
            if loss_terms is not None:
                epoch_base_loss_sum += float(loss_terms.get('base_loss', 0.0))
                epoch_div_loss_sum += float(loss_terms.get('div_loss', 0.0))
                epoch_dpp_loss_sum += float(loss_terms.get('dpp_loss', 0.0))
                epoch_final_loss_sum += float(loss_terms.get('final_loss', loss_val))
            else:
                epoch_base_loss_sum += loss_val
                epoch_final_loss_sum += loss_val

            if t_args.log_interval > 0 and (num % (t_args.log_interval * 10)) == 0:
                if loss_terms is not None:
                    print(
                        f"[Train][Epoch {i+1}/{max_epoch}][Iter {num}] "
                        f"base_loss={loss_terms.get('base_loss', 0.0):.6f} "
                        f"div_loss={loss_terms.get('div_loss', 0.0):.6f} "
                        f"dpp_loss={loss_terms.get('dpp_loss', 0.0):.6f} "
                        f"final_loss={loss_terms.get('final_loss', loss_val):.6f}"
                    )
                else:
                    print(f"[Train][Epoch {i+1}/{max_epoch}][Iter {num}] loss={loss_val:.6f}")

            if (num + 1) % t_args.eval_batch == 0:
                prorec.eval()
                recall_1, recall_10, recall_50, f1_results, coverage_results = evaluate_rec_redial(
                    test_loader, prorec, graph_data, args)
                stats_all['recall_1'].append(recall_1)
                stats_all['recall_10'].append(recall_10)
                stats_all['recall_50'].append(recall_50)
                stats_all['f1@1'].append(f1_results['f1@1'])
                stats_all['f1@10'].append(f1_results['f1@10'])
                stats_all['f1@50'].append(f1_results['f1@50'])
                for cov_key, cov_val in coverage_results.items():
                    if cov_key in stats_all:
                        stats_all[cov_key].append(cov_val)

                if recall_1 > best_recall_1:
                    best_recall_1 = recall_1
                    print("saving model...")
                    torch.save(prorec.state_dict(), save_path_1)
                    torch.save(prorec.state_dict(), save_path)
                if recall_10 > best_recall_10:
                    best_recall_10 = recall_10
                    print("saving model...")
                    torch.save(prorec.state_dict(), save_path_10)
                if recall_50 > best_recall_50:
                    best_recall_50 = recall_50
                    print("saving model...")
                    torch.save(prorec.state_dict(), save_path_50)

                if f1_results['f1@1'] > best_f1_1:
                    best_f1_1 = f1_results['f1@1']
                    print(colored('f1@1 new high, saving model...','green'))
                    torch.save(prorec.state_dict(), save_path_f1_1)

                if f1_results['f1@10'] > best_f1_10:
                    best_f1_10 = f1_results['f1@10']
                    print(colored('f1@10 new high, saving model...','green'))
                    torch.save(prorec.state_dict(), save_path_f1_10)

                if f1_results['f1@50'] > best_f1_50:
                    best_f1_50 = f1_results['f1@50']
                    print(colored('f1@50 new high, saving model...','green'))
                    torch.save(prorec.state_dict(), save_path_f1_50)

                if coverage_results['item_coverage@10'] > best_coverage_10:
                    best_coverage_10 = coverage_results['item_coverage@10']
                    print(colored('item_coverage@10 new high, saving model...','green'))
                    torch.save(prorec.state_dict(), save_path_cov10)

                if coverage_results['item_coverage@1'] > best_coverage_1:
                    best_coverage_1 = coverage_results['item_coverage@1']
                    print(colored('item_coverage@1 new high, saving model...','green'))
                    torch.save(prorec.state_dict(), save_path_cov1)

                if coverage_results['item_coverage@50'] > best_coverage_50:
                    best_coverage_50 = coverage_results['item_coverage@50']
                    print(colored('item_coverage@50 new high, saving model...','green'))
                    torch.save(prorec.state_dict(), save_path_cov50)
                prorec.train()
                with open('stats_' + model_name + '.json', 'w') as f:
                    json.dump(stats_all, f)
            num += 1

        epoch_train_seconds = time.time() - epoch_start_time
        epoch_avg_loss = epoch_loss_sum / max(epoch_steps, 1)
        epoch_avg_base_loss = epoch_base_loss_sum / max(epoch_steps, 1)
        epoch_avg_div_loss = epoch_div_loss_sum / max(epoch_steps, 1)
        epoch_avg_dpp_loss = epoch_dpp_loss_sum / max(epoch_steps, 1)
        epoch_avg_final_loss = epoch_final_loss_sum / max(epoch_steps, 1)
        print(f"[Epoch End] {i+1}/{max_epoch}  avg_train_loss={epoch_avg_loss:.6f}  "
              f"epoch_train_time={epoch_train_seconds:.2f}s")
        print(f"[Epoch Loss] {i+1}/{max_epoch}  base_loss={epoch_avg_base_loss:.6f}  "
              f"div_loss={epoch_avg_div_loss:.6f}  dpp_loss={epoch_avg_dpp_loss:.6f}  "
              f"final_loss={epoch_avg_final_loss:.6f}")

        if prev_epoch_avg_loss is not None:
            delta = prev_epoch_avg_loss - epoch_avg_loss
            if delta < 0:
                print("[Warn][Underfit/Stall] Train loss increased this epoch.")
            elif delta < 1e-4:
                print("[Info][Slow] Train loss barely decreased.")
        prev_epoch_avg_loss = epoch_avg_loss

        prorec.eval()
        recall_1_ep, recall_10_ep, recall_50_ep, f1_results_ep, coverage_results_ep = evaluate_rec_redial(
            test_loader, prorec, graph_data, args)
        print(f"[Epoch Eval] epoch={i+1}  recall@1={recall_1_ep:.6f}  "
              f"recall@10={recall_10_ep:.6f}  recall@50={recall_50_ep:.6f}")

        if f1_results_ep['f1@1'] > best_f1_1:
            best_f1_1 = f1_results_ep['f1@1']
            print(colored('f1@1 new high (epoch eval), saving model...','green'))
            torch.save(prorec.state_dict(), save_path_f1_1)

        if f1_results_ep['f1@10'] > best_f1_10:
            best_f1_10 = f1_results_ep['f1@10']
            print(colored('f1@10 new high (epoch eval), saving model...','green'))
            torch.save(prorec.state_dict(), save_path_f1_10)

        if f1_results_ep['f1@50'] > best_f1_50:
            best_f1_50 = f1_results_ep['f1@50']
            print(colored('f1@50 new high (epoch eval), saving model...','green'))
            torch.save(prorec.state_dict(), save_path_f1_50)

        if coverage_results_ep['item_coverage@10'] > best_coverage_10:
            best_coverage_10 = coverage_results_ep['item_coverage@10']
            print(colored('item_coverage@10 new high (epoch eval), saving model...','green'))
            torch.save(prorec.state_dict(), save_path_cov10)

        if coverage_results_ep['item_coverage@1'] > best_coverage_1:
            best_coverage_1 = coverage_results_ep['item_coverage@1']
            print(colored('item_coverage@1 new high (epoch eval), saving model...','green'))
            torch.save(prorec.state_dict(), save_path_cov1)

        if coverage_results_ep['item_coverage@50'] > best_coverage_50:
            best_coverage_50 = coverage_results_ep['item_coverage@50']
            print(colored('item_coverage@50 new high (epoch eval), saving model...','green'))
            torch.save(prorec.state_dict(), save_path_cov50)

        if recall_10_ep > best_epoch_recall10:
            best_epoch_recall10 = recall_10_ep
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"[EarlyStop] No epoch-level improvement: "
                  f"{no_improve_epochs}/{t_args.early_stop_patience}")
            if no_improve_epochs >= t_args.early_stop_patience:
                print(f"[EarlyStop] Stopping at epoch {i+1}.")
                break

elif option == "test":
    print("testing model recommendation...")
    state_dict = torch.load(save_path, map_location=device_str)
    prorec = ProRec(
        device_str=device_str,
        dataset="tgredial",
        graph_embed_size=t_args.graph_embed_size,
        utter_embed_size=t_args.utter_embed_size,
        negative_sample_ratio=t_args.negative_sample_ratio,
        word_net=t_args.word_net,
        div_loss_weight=t_args.div_loss_weight,
        div_temperature=t_args.div_temperature,
        dpp_loss_weight=t_args.dpp_loss_weight,
    )
    prorec.load_state_dict(state_dict, strict=False)
    prorec.eval()
    prorec.to(device)
    evaluate_rec_redial(test_loader, prorec, graph_data, args)

elif option == "test_gen":
    print("testing model generation...")
    state_dict = torch.load(save_path, map_location=device_str)
    prorec = ProRec(
        device_str=device_str,
        dataset="tgredial",
        graph_embed_size=t_args.graph_embed_size,
        utter_embed_size=t_args.utter_embed_size,
        negative_sample_ratio=t_args.negative_sample_ratio,
        word_net=t_args.word_net,
        div_loss_weight=t_args.div_loss_weight,
        div_temperature=t_args.div_temperature,
        dpp_loss_weight=t_args.dpp_loss_weight,
    )
    prorec.load_state_dict(state_dict, strict=False)
    prorec.eval()
    prorec.to(device)
    evaluate_gen_redial(test_loader, prorec, graph_data, args, golden_intent=False)
