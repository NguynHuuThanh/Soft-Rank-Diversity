import json
import numpy as np
import torch
import random
import sys
import os.path as osp
from tqdm import tqdm
from termcolor import colored
from copy import deepcopy
import math
import random

from CR_walker import ProRec
from torch_geometric.loader import DataLoader

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from data.redial import ReDial
from data.gorecdial import GoRecDial
from data.metrics import bleu,f1_score,distinct_n_grams
from data.utils import da_tree_serial,utter_lexical_redial,utter_lexical_gorecdial


# ───────────────────────────────────────────────────────
# Coverage metric helpers
# ───────────────────────────────────────────────────────

def compute_kg_coverage(topk_item_ids, movie_kg_neighbors, total_reachable_entities):
    """
    KG-Entity Coverage@k: fraction of reachable KG entities covered by
    the 1-hop neighborhoods of the top-k items.
    """
    covered = set()
    for item_id in topk_item_ids:
        covered |= movie_kg_neighbors.get(item_id, set())
    return len(covered) / total_reachable_entities


def init_coverage_accumulators(k_values):
    """Return fresh per-turn and per-dialog accumulator dicts for all k."""
    acc = {}
    for k in k_values:
        acc[f'kg_cov_turn@{k}'] = []
        acc[f'kg_cov_dialog@{k}'] = []
    return acc


def accumulate_turn_coverage(acc, scores_array, item_ids, k_values, args):
    """
    Given a score array and corresponding item id list, compute coverage
    for each k and append to the per-turn accumulator lists.
    Returns the per-turn kg/cat values for the current turn (for dialog agg).
    """
    movie_kg_neighbors = args['movie_kg_neighbors']
    total_reachable = args['total_reachable_entities']

    # Rank item_ids by descending score
    ranked_indices = sorted(range(len(scores_array)), key=lambda x: -scores_array[x])
    ranked_items = [item_ids[i] for i in ranked_indices]

    turn_vals = {}
    for k in k_values:
        topk = ranked_items[:k]
        kg = compute_kg_coverage(topk, movie_kg_neighbors, total_reachable)
        acc[f'kg_cov_turn@{k}'].append(kg)
        turn_vals[k] = kg
    return turn_vals


def finalize_coverage(acc, k_values):
    """Average all accumulator lists and return a flat results dict."""
    results = {}
    for k in k_values:
        for metric in ['kg_cov_turn', 'kg_cov_dialog']:
            key = f'{metric}@{k}'
            vals = acc[key]
            results[key] = sum(vals) / max(len(vals), 1)
    return results


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


def compute_item_coverage_redial(all_scores_list, n_movies):
    """
    Item Coverage@1, Item Coverage@10 and Item Coverage@50 for ReDial.
    Coverage@k = fraction of catalog items that appear at least once in
    any sample's top-k recommendation list.
    """
    all_top1_items = set()
    all_top10_items = set()
    all_top50_items = set()

    for scores in all_scores_list:
        if len(scores) != n_movies:
            continue
        top50_idx = np.argsort(scores)[-50:]
        top1_idx = top50_idx[-1:]
        top10_idx = top50_idx[-10:]
        all_top1_items.update(top1_idx.tolist())
        all_top10_items.update(top10_idx.tolist())
        all_top50_items.update(top50_idx.tolist())

    if n_movies > 0:
        coverage_1 = len(all_top1_items) / n_movies
        coverage_10 = len(all_top10_items) / n_movies
        coverage_50 = len(all_top50_items) / n_movies
    else:
        coverage_1 = 0
        coverage_10 = 0
        coverage_50 = 0

    return {
        'item_coverage@1': coverage_1,
        'item_coverage@10': coverage_10,
        'item_coverage@50': coverage_50
    }


def select_intent(sel_intent,mentioned,args):
    device=torch.device('cuda:0')
    attribute_types=set([ 'Person', 'Time', 'Genre', 'Subject'])
    last=[]
    bat=[]  
    sel=[]
    grp=[]
    #itt=[]
    grp_bat=[]
    mask=[]
    node_candidate1=[]
    cur_grp_size=sel_intent.size()[0]
    for i in range(cur_grp_size):
        cand=[]
        last.append(args['none_node'])
        grp_bat.append(i)
        #itt.append(sel_intent[i])
        if sel_intent[i]==0:#chat
            for item in mentioned[i]:
                cand.append(item)
                sel.append(item)
                bat.append(i)
                grp.append(i)
                mask.append(1)
            if len(mentioned[i])==0:
                cand.append(args['none_node'])
                sel.append(args['none_node'])
                bat.append(i)
                grp.append(i)
                mask.append(1)
        elif sel_intent[i]==1:#question
            #print(args['generals'])
            for item in args['generals']:
                cand.append(item)
                sel.append(item)
                bat.append(i)
                grp.append(i)
                mask.append(1)
        else: #recommend
            my_cand=set()
            for item in mentioned[i]:
                if item<args['movie_count']:
                    my_cand=my_cand.union(args['attribute_dict'][item])
                elif args['nodes'][item]['type'] in attribute_types:
                    my_cand.add(item)
            cand=list(my_cand)
            if len(cand)==0:
                cand.append(args['none_node'])
            for item in cand:
                sel.append(item)
                bat.append(i)
                grp.append(i)
                mask.append(1)
        node_candidate1.append(cand)
    sel_index=torch.Tensor(sel).long().to(device=device)
    grp_index=torch.Tensor(grp).long().to(device=device)
    batch_index=torch.Tensor(bat).long().to(device=device)
    score_mask=torch.Tensor(mask).float().to(device=device)
    #print(intent_index)
    grp_bat_index=torch.Tensor(grp_bat).long().to(device=device)
    last_index=torch.Tensor(last).long().to(device=device)
    return sel_index,grp_index,batch_index,grp_bat_index,last_index,node_candidate1,score_mask


def select_layer_1(nodes,step,step_grp,intent_label,node_candidate1,label_1,mentioned,args,rec_cand=None,dataset="redial"):
    device=torch.device('cuda:0')
    cur_grp_size=intent_label.size()[0]
    cand_num=[0 for _ in range(cur_grp_size+1)]
    all_selected=[]
    for i in step_grp:
        cand_num[i]+=1
    start=0
    end=cand_num[0]

    last=[]
    bat=[]  
    sel=[]
    grp=[]
    itt=[]
    grp_bat=[]
    mask=[]

    grp_num=0
    all_candidates=[]

    all_split=[]
    for num in range(cur_grp_size):
        selected=[]
        split_score=[]
        score=step[start:end].tolist()
        if len(node_candidate1[num])!=0:
            best_idx=score.index(max(score))
            if intent_label[num]==0:
                for nod,scr in enumerate(score):
                    if scr>args['threshold'][0][0]:
                        last.append(node_candidate1[num][nod])
                        selected.append(node_candidate1[num][nod])
                        if nodes[node_candidate1[num][nod]]['type']=='Movie':
                            node_candidate2=list(args['attribute_dict'][node_candidate1[num][nod]])+[args['none_node']]
                        else:
                            node_candidate2=[args['none_node']]
                        all_candidates.append(node_candidate2)
                        sel=sel+node_candidate2
                        bat=bat+[num for _ in range(len(node_candidate2))]
                        grp=grp+[grp_num for _ in range(len(node_candidate2))]
                        mask=mask+[1 for _ in range(len(node_candidate2))]
                        grp_bat.append(num)
                        itt.append(intent_label[num])
                        grp_num+=1
            elif intent_label[num]==1:
                step_general_dict=deepcopy(args['generals_dict'])
                #print(step_general_dict)
                for n in mentioned[num]:
                    if nodes[n]['type']!="Attr" and nodes[n]['type']!="None":
                        if nodes[n]['type']=="Person":
                            for rol in nodes[n]['role']:
                                step_general_dict[rol].add(nodes[n]['global'])
                        else:
                            step_general_dict[nodes[n]['type']].add(nodes[n]['global'])
                        if nodes[n]['type']=="Movie":
                            for attr in args['attribute_dict'][nodes[n]['global']]:
                                if nodes[attr]['type']!="Attr":
                                    if nodes[attr]['type']=="Person":
                                        for rol in nodes[attr]['role']:
                                            step_general_dict[rol].add(attr)
                                    else:
                                        step_general_dict[nodes[attr]['type']].add(attr)
                #print(step_general_dict)
                for nod,scr in enumerate(score):
                    if scr>args['threshold'][0][1]:
                        selected.append(node_candidate1[num][nod])
                        last.append(node_candidate1[num][nod])
                        cur_gen=nodes[node_candidate1[num][nod]]['name']
                        node_candidate2=list(step_general_dict[cur_gen])+[args['none_node']]
                        all_candidates.append(node_candidate2)
                        sel=sel+node_candidate2
                        bat=bat+[num for _ in range(len(node_candidate2))]
                        grp=grp+[grp_num for _ in range(len(node_candidate2))]
                        mask=mask+[1 for _ in range(len(node_candidate2)) ]
                        grp_bat.append(num)
                        itt.append(intent_label[num])
                        grp_num+=1
            else:
                if args['sample']>0:
                    score_torch=torch.sigmoid(torch.Tensor(score))
                    top_k=score_torch.topk(min(args['sample'],len(score))).indices.numpy()
                    top_scores=score_torch.topk(min(args['sample'],len(score))).values.numpy()
                    #print(top_scores)
                    split_score.append(top_scores)
                    threshold=1 / (1 + math.exp(-args['threshold'][0][2]))
                    if top_scores[0]>threshold:
                        for nod, top_idx in enumerate(top_k):
                            if top_scores[nod]>threshold:
                                attr_idx=node_candidate1[num][top_idx]
                                selected.append(attr_idx)
                                last.append(attr_idx)
                                if dataset=="redial":
                                    node_candidate2=[mov for mov in range(args['movie_count'])]
                                    my_mask=[0 for mov in range(args['movie_count'])]
                                    # print(attr_idx)
                                    # print(args['attribute_dict'][attr_idx])
                                    for idx in args['attribute_dict'][attr_idx]:
                                        my_mask[idx]=1
                                else:
                                    node_candidate2=rec_cand[num]
                                    my_mask=[1 for mov in range(len(node_candidate2))]

                                all_candidates.append(node_candidate2)
                                sel=sel+node_candidate2
                                bat=bat+[num for _ in range(len(node_candidate2))]
                                grp=grp+[grp_num for _ in range(len(node_candidate2))]
                                mask=mask+my_mask
                                grp_bat.append(num)
                                itt.append(intent_label[num])
                                grp_num+=1
                    else:
                        attr_idx=node_candidate1[num][top_k[0]]
                        selected.append(attr_idx)
                        last.append(attr_idx)
                        if dataset=="redial":
                            node_candidate2=[mov for mov in range(args['movie_count'])]
                            my_mask=[0 for mov in range(args['movie_count'])]
                            # print(attr_idx)
                            # print(args['attribute_dict'][attr_idx])
                            for idx in args['attribute_dict'][attr_idx]:
                                my_mask[idx]=1
                        else:
                            node_candidate2=rec_cand[num]
                            my_mask=[1 for mov in range(len(node_candidate2))]

                        all_candidates.append(node_candidate2)
                        sel=sel+node_candidate2
                        bat=bat+[num for _ in range(len(node_candidate2))]
                        grp=grp+[grp_num for _ in range(len(node_candidate2))]
                        mask=mask+my_mask
                        grp_bat.append(num)
                        itt.append(intent_label[num])
                        grp_num+=1

                else:
                    for nod,scr in enumerate(score):
                        if scr>args['threshold'][0][2]:
                            last.append(node_candidate1[num][nod])
                            selected.append(node_candidate1[num][nod])
                            if dataset=="redial":
                                node_candidate2=[mov for mov in range(args['movie_count'])]
                                my_mask=[0 for mov in range(args['movie_count'])]
                                for idx in args['attribute_dict'][node_candidate1[num][nod]]:
                                    my_mask[idx]=1
                            else:
                                node_candidate2=rec_cand[num]
                                my_mask=[1 for mov in range(len(node_candidate2))]
                            all_candidates.append(node_candidate2)
                            sel=sel+node_candidate2
                            bat=bat+[num for _ in range(len(node_candidate2))]
                            grp=grp+[grp_num for _ in range(len(node_candidate2))]
                            mask=mask+my_mask
                            grp_bat.append(num)
                            itt.append(intent_label[num])
                            grp_num+=1
        all_selected.append(selected)
        all_split.append(split_score)
        
        start+=cand_num[num]
        end+=cand_num[num+1]
    sel_index=torch.Tensor(sel).long().to(device=device)
    grp_index=torch.Tensor(grp).long().to(device=device)
    batch_index=torch.Tensor(bat).long().to(device=device)
    intent_index=torch.Tensor(itt).long().to(device=device)
    grp_bat_index=torch.Tensor(grp_bat).long().to(device=device)
    last_index=torch.Tensor(last).long().to(device=device)
    score_mask=torch.Tensor(mask).float().to(device=device)

    return sel_index,grp_index,batch_index,intent_index,grp_bat_index,last_index,score_mask,all_candidates,all_selected,all_split
            

def select_layer_2(step,step_grp,grp_batch,intent_label,node_candidate2,batch_size,args):
    cur_grp_size=step_grp[-1]+1
    cand_num=[0 for _ in range(cur_grp_size+1)]
    for i in step_grp:
        cand_num[i]+=1
    start=0
    end=cand_num[0]
    selected=[[] for _ in range(batch_size)]
    for num in range(cur_grp_size):
        score=step[start:end].tolist()
        threshold=args['threshold'][1][intent_label[num]]
        k=0
        cur_sel=[]
        for i,scr in enumerate(score):
            if scr>threshold and i!=len(score)-1:
                cur_sel.append(node_candidate2[num][i])
                k+=1
            if k==args['max_leaf']:
                break
        selected[grp_batch[num]].append(cur_sel)
        #index=score.index(max(score))
        #selected[grp_batch[num]].append([node_candidate2[num][index]])
        start+=cand_num[num]
        end+=cand_num[num+1]

    return selected

        
def evaluate_rec_redial(test_loader:DataLoader, model:ProRec,graph_data,args,eval_batch=None):

    hit_1=0
    hit_10=0
    hit_50=0
    intent_accuracy=0
    da_distrib=[0,0,0]
    tot_rec=0
    recommend_turns=0
    tot=0
    
    # Coverage accumulators
    k_values = args.get('coverage_topk', [1, 10, 50])
    cov_acc = init_coverage_accumulators(k_values)
    # Per-dialog tracking: accumulate item ids recommended within a dialog
    dialog_rec_items = []  # flat list of movie ids recommended so far in current dialog
    # For Item Coverage@k across all recommend samples
    all_scores_list = []

    batches=0
    model.eval()

    generated_DAs=[]

    with torch.no_grad():
        for test_batch in tqdm(test_loader):
            tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,intent_label,label_1,label_2,score_masks,word_index,word_batch_index=model.prepare_data_redial(test_batch.dialog_history,test_batch.mention_history,test_batch.intent,test_batch.node_candidate1,test_batch.node_candidate2,graph_data.edge_type,graph_data.edge_index,test_batch.label_1,test_batch.label_2,test_batch.gold_pos,args['attribute_dict'])

            intent=model.get_intent(tokenized_dialog,all_length,maxlen,init_hidden)
            selected=intent.max(dim=-1).indices
            step1,last_weight1,partial_score_1=model.inference_redial(intent_indices[0],tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_indices[0],sel_batch_indices[0],sel_group_indices[0],grp_batch_indices[0],last_indices[0],score_masks[0],word_index,word_batch_index)
            # rec=model.get_rec(tokenized_dialog,edge_type,edge_index,node_feature,)
            cur_batch_size=intent.size()[0]

            step1=step1.cpu().numpy()
            step_grp=sel_group_indices[0].cpu().numpy()
            # for i in step_grp:
            #     cand_num[i]+=1s
            sel_index_2,grp_index_2,batch_index_2,intent_index_2,grp_bat_index_2,last_index_2,score_mask_2,node_candidate2,selected_1,all_split=select_layer_1(args['nodes'],step1,step_grp,intent_label,test_batch.node_candidate1,test_batch.label_1,test_batch.mention_history,args)
            sel_grp_size=grp_bat_index_2.size()[0]
            step2,_,partial_score_2=model.inference_redial(intent_index_2,tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_index_2,batch_index_2,grp_index_2,grp_bat_index_2,last_index_2,score_mask_2,word_index,word_batch_index,last_weights=last_weight1,layer=1)

            cand_num=[0 for _ in range(sel_grp_size+1)]
            step_grp=grp_index_2.cpu().numpy()
            grp_batch=grp_bat_index_2.cpu().numpy()
            step2=step2.cpu().numpy()
            #partial_score_2=partial_score_2.cpu().numpy()
            intent_label_2=intent_index_2.cpu().numpy()


            for i in step_grp:
                cand_num[i]+=1

            start=0
            end=cand_num[0]
            all_scores=[]
            flags=[0 for _ in range(cur_batch_size)]

            for it in test_batch.intent:
                if it=="recommend":
                    all_scores.append(np.array([0 for _ in range(args['movie_count'])]))
                else:
                    all_scores.append([])

            #print(test_batch.intent)


            for num in range(sel_grp_size):
                my_label=test_batch.label_rec[grp_batch[num]]
                if intent_label_2[num]==2:
                    score=step2[start:end]
                    all_scores[grp_batch[num]]=all_scores[grp_batch[num]]+score#*all_split[grp_batch[num]][0][flags[grp_batch[num]]]
                    flags[grp_batch[num]]+=1
                start+=cand_num[num]
                end+=cand_num[num+1]


            
            for num in range(cur_batch_size):
                my_label=test_batch.label_rec[num]
                wrong_count=[0 for _ in range(len(my_label))]
                if test_batch.intent[num]=="recommend":
                    recommend_turns += 1
                    # item_ids for ReDial are movie indices 0..movie_count-1
                    item_ids = list(range(args['movie_count']))
                    scores_arr = all_scores[num]
                    all_scores_list.append(scores_arr)
                    # Per-turn coverage
                    turn_vals = accumulate_turn_coverage(cov_acc, scores_arr, item_ids, k_values, args)
                    # Collect top-k items for dialog-level coverage (use max k)
                    max_k = max(k_values)
                    ranked = sorted(range(len(scores_arr)), key=lambda x: -scores_arr[x])
                    dialog_rec_items.extend([item_ids[i] for i in ranked[:max_k]])

                    for item in all_scores[num]:
                        for p,idx in enumerate(my_label):
                            if item>=all_scores[num][idx]:
                                wrong_count[p]+=1
                    for item in wrong_count:
                        tot_rec+=1
                        if item<2:
                            hit_1+=1
                        if item<11:
                            hit_10+=1
                        if item<51:
                            hit_50+=1

                # Dialog boundary: flush dialog-level coverage
                if test_batch.last_turn[num]==1 and len(dialog_rec_items) > 0:
                    for k in k_values:
                        topk_dialog = dialog_rec_items[:k] if len(dialog_rec_items) >= k else dialog_rec_items
                        kg = compute_kg_coverage(topk_dialog, args['movie_kg_neighbors'], args['total_reachable_entities'])
                        cov_acc[f'kg_cov_dialog@{k}'].append(kg)
                    dialog_rec_items = []

            if batches==eval_batch:
                break
            batches+=1
            tot+=cur_batch_size

    recall_1=hit_1/tot_rec if tot_rec > 0 else 0.0
    recall_10=hit_10/tot_rec if tot_rec > 0 else 0.0
    recall_50=hit_50/tot_rec if tot_rec > 0 else 0.0

    precision_1 = hit_1 / max(recommend_turns * 1, 1)
    precision_10 = hit_10 / max(recommend_turns * 10, 1)
    precision_50 = hit_50 / max(recommend_turns * 50, 1)

    def _f1(precision, recall):
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    f1_1 = _f1(precision_1, recall_1)
    f1_10 = _f1(precision_10, recall_10)
    f1_50 = _f1(precision_50, recall_50)
    f1_results = {
        'f1@1': f1_1,
        'f1@10': f1_10,
        'f1@50': f1_50,
    }

    coverage_results = finalize_coverage(cov_acc, k_values)
    item_coverage_results = compute_item_coverage_redial(all_scores_list, args['movie_count'])
    coverage_results.update(item_coverage_results)

    print("recall_1",recall_1)
    print('recall_10:',recall_10)
    print("recall_50:",recall_50)
    print("f1@1:", f1_1)
    print("f1@10:", f1_10)
    print("f1@50:", f1_50)
    for key, val in sorted(coverage_results.items()):
        print(f"{key}: {val:.4f}")

    return recall_1,recall_10,recall_50,f1_results,coverage_results
    

def evaluate_rec_gorecdial(test_loader:DataLoader, model:ProRec,graph_data, bow_data,args,eval_batch=None):
    correct=0
    da_distrib=[0,0,0]
    

    intent_accuracy=0

    tot_turns=0
    tot_dialog=0

    turn_1=0
    turn_3=0
    chat_1=0
    chat_3=0

    turn_1_ex=0
    turn_3_ex=0
    chat_1_ex=0
    chat_3_ex=0

    splits=6
    accuracy_split=[0 for _ in range(splits)]
    tot_split=[0 for _ in range(splits)]
    correct=[]
    turn=0
    generated_DAs=[]

    # Coverage accumulators
    k_values = args.get('coverage_topk', [1, 10, 50])
    cov_acc = init_coverage_accumulators(k_values)
    dialog_rec_items = []  # accumulate recommended item ids within a dialog
    all_rec_lists = []
    with torch.no_grad():
        for test_batch in tqdm(test_loader):

            tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,bow_embed,rec_index,rec_batch_index,rec_golden,intent_label,label_1,label_2,score_masks,word_index,word_batch_index=model.prepare_data_gorecdial(test_batch.dialog_history,test_batch.mention_history,test_batch.intent,test_batch.node_candidate1,test_batch.node_candidate2,graph_data.edge_type,graph_data.edge_index,test_batch.label_c,test_batch.label_2,args['attribute_dict'],test_batch.rec_cand,bow_data.bow_embed)
            intent=model.get_intent(tokenized_dialog,all_length,maxlen,init_hidden)


            
            #rec_intent,rec_index_ex,rec_batch_index_ex,shuffled_rec_cand,golden=model.prepare_rectest(test_batch.rec_cand)
            rec_intent,rec_index_ex,rec_batch_index_ex,shuffled_rec_cand,golden=model.prepare_rectest(test_batch.rec_cand)
            #print(shuffled_rec_cand)
            rec_index,rec_group_index,rec_batch_index,rec_grp_bat_index,rec_last_index,rec_node_candidate1,rec_score_mask=select_intent(rec_intent,test_batch.mention_history,args)

            
            step1,last_weight1,partial_score,rec=model.inference_gorecdial(rec_intent,tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,rec_index,rec_batch_index,rec_group_index,rec_grp_bat_index,rec_last_index,rec_score_mask,word_index,word_batch_index,bow_embed,rec_index_ex,rec_batch_index_ex)

            step1=step1.cpu().detach().numpy()
            step_grp=rec_group_index.cpu().numpy()
            

            sel_index_2,grp_index_2,batch_index_2,intent_index_2,grp_bat_index_2,last_index_2,score_mask_2,node_candidate2,selected_1,all_split=select_layer_1(args['nodes'],step1,step_grp,rec_intent,rec_node_candidate1,test_batch.label_1,test_batch.mention_history,args,shuffled_rec_cand,"gorecdial")
            sel_grp_size=grp_bat_index_2.size()[0]
            grp_batch=grp_bat_index_2.cpu().numpy()


            step2,_,partial_score_2=model.inference_gorecdial(intent_index_2,tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_index_2,batch_index_2,grp_index_2,grp_bat_index_2,last_index_2,score_mask_2,word_index,word_batch_index,last_weights=last_weight1,layer=1)

            step2=step2.cpu().numpy()
            intent_label_2=intent_index_2.cpu().numpy()

            start=0
            all_scores=[]

            for it in rec_intent:
                all_scores.append(np.array([0 for _ in range(5)]))

            for num in range(sel_grp_size):
                score=step2[start:start+5]
                all_scores[grp_batch[num]]=all_scores[grp_batch[num]]+score
                start+=5
            

            cur_batch_size=intent.size()[0]
            for num in range(cur_batch_size):
                score=all_scores[num]
                score_ex=rec[num,:]

                # --- Per-turn coverage from score_ex (explicit recommender) ---
                score_ex_np = score_ex.cpu().numpy() if torch.is_tensor(score_ex) else np.array(score_ex)
                cand_ids = shuffled_rec_cand[num]  # list of 5 movie ids
                all_rec_lists.append(cand_ids)
                turn_vals = accumulate_turn_coverage(cov_acc, score_ex_np, cand_ids, k_values, args)
                # Collect top items for dialog-level coverage
                max_k = max(k_values)
                ranked_idx = sorted(range(len(score_ex_np)), key=lambda x: -score_ex_np[x])
                dialog_rec_items.extend([cand_ids[i] for i in ranked_idx[:max_k]])

                wrong_count=0
                wrong_count_ex=0
                for item in score:
                    if item>=score[golden[num]]:
                        wrong_count+=1
                
                for item in score_ex:
                    if item>=score_ex[golden[num]]:
                        wrong_count_ex+=1
                
                tot_turns+=1
                if wrong_count<2:
                    turn_1+=1
                if wrong_count<4:
                    turn_3+=1

                if wrong_count_ex<2:
                    correct.append(1)
                    turn_1_ex+=1
                else:
                    correct.append(0)
                if wrong_count_ex<4:
                    turn_3_ex+=1

                if test_batch.last_turn[num]==1:
                    tot_dialog+=1
                    if wrong_count<2:
                        chat_1+=1
                    if wrong_count<4:
                        chat_3+=1
                    
                    if wrong_count_ex<2:
                        chat_1_ex+=1
                    if wrong_count_ex<4:
                        chat_3_ex+=1
                    

                    for i,item in enumerate(correct):
                        index=int((i/(turn+1e-5))*splits)
                        tot_split[index]+=1
                        if item==1:
                            accuracy_split[index]+=1
                    turn=0
                    correct=[]

                    # Dialog-level coverage
                    if len(dialog_rec_items) > 0:
                        for k in k_values:
                            topk_dialog = dialog_rec_items[:k] if len(dialog_rec_items) >= k else dialog_rec_items
                            kg = compute_kg_coverage(topk_dialog, args['movie_kg_neighbors'], args['total_reachable_entities'])
                            cov_acc[f'kg_cov_dialog@{k}'].append(kg)
                        dialog_rec_items = []
                else:
                    turn+=1
    
    turn_1=turn_1/tot_turns
    turn_3=turn_3/tot_turns
    chat_1=chat_1/tot_dialog
    chat_3=chat_3/tot_dialog

    turn_1_ex=turn_1_ex/tot_turns
    turn_3_ex=turn_3_ex/tot_turns
    chat_1_ex=chat_1_ex/tot_dialog
    chat_3_ex=chat_3_ex/tot_dialog

    for i,item in enumerate(accuracy_split):
        accuracy_split[i]=item/tot_split[i]

    coverage_results = finalize_coverage(cov_acc, k_values)
    item_coverage_results = compute_item_coverage_gorecdial(all_rec_lists, args['movie_count'])
    coverage_results.update(item_coverage_results)

    intent_accuracy=intent_accuracy/tot_turns
    print("turn_1",turn_1)
    print("turn_3",turn_3)
    print("chat_1",chat_1)
    print("chat_3",chat_3)
    print("turn_1_ex",turn_1_ex)
    print("turn_3_ex",turn_3_ex)
    print("chat_1_ex",chat_1_ex)
    print("chat_3_ex",chat_3_ex)
    for key, val in sorted(coverage_results.items()):
        print(f"{key}: {val:.4f}")
        
    return intent_accuracy,turn_1,turn_3,chat_1,chat_3,turn_1_ex,turn_3_ex,chat_1_ex,chat_3_ex,coverage_results


def evaluate_gen_redial(test_loader:DataLoader, model:ProRec, graph_data, args, golden_intent=True):
    model.eval()
    generated_DAs=[]
    generated_utters=[]
    all_gpt_in=[]
    cnt=0
    from generator import Generator
    gener=Generator(args['gen_conf'])
    with torch.no_grad():
        for test_batch in tqdm(test_loader):
            tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,intent_label,label_1,label_2,score_masks,word_index,word_batch_index=model.prepare_data_redial(test_batch.dialog_history,test_batch.mention_history,test_batch.intent,test_batch.node_candidate1,test_batch.node_candidate2,graph_data.edge_type,graph_data.edge_index,test_batch.label_1,test_batch.label_2,test_batch.gold_pos,args['attribute_dict'])


            intent=model.get_intent(tokenized_dialog,all_length,maxlen,init_hidden)
            selected=intent.max(dim=-1).indices
            cur_batch_size=intent.size()[0]
            
            
            if golden_intent:
                step1,last_weight1,_=model.inference_redial(intent_indices[0],tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_indices[0],sel_batch_indices[0],sel_group_indices[0],grp_batch_indices[0],last_indices[0],score_masks,word_index,word_batch_index,layer=0)
                step1=step1.cpu().numpy()
                step_grp=sel_group_indices[0].cpu().numpy()
                
                sel_index_2,grp_index_2,batch_index_2,intent_index_2,grp_bat_index_2,last_index_2,score_mask_2,node_candidate2,selected_1,all_split=select_layer_1(args['nodes'],step1,step_grp,intent_label,test_batch.node_candidate1,test_batch.label_1,test_batch.mention_history,args)
                step2,_,_=model.inference_redial(intent_index_2,tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_index_2,batch_index_2,grp_index_2,grp_bat_index_2,last_index_2,score_mask_2,word_index,word_batch_index,last_weights=last_weight1,layer=1)

                selected_2=select_layer_2(step2,grp_index_2.cpu().numpy(),grp_bat_index_2.cpu().numpy(),intent_index_2.cpu().numpy(),node_candidate2,cur_batch_size,args)
                
            else:
                sel_index_1i,grp_index_1i,batch_index_1i,grp_bat_index_1i,last_index_1i,node_candidate1i,score_mask1i=select_intent(selected,test_batch.mention_history,args)
                step1i,last_weight1i,_=model.inference_redial(selected,tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_index_1i,batch_index_1i,grp_index_1i,grp_bat_index_1i,last_index_1i,score_mask1i,word_index,word_batch_index)
                step1i=step1i.cpu().numpy()
                step_grp=grp_index_1i.cpu().numpy()

                sel_index_2,grp_index_2,batch_index_2,intent_index_2,grp_bat_index_2,last_index_2,score_mask_2,node_candidate2,selected_1,all_split=select_layer_1(args['nodes'],step1i,step_grp,selected,node_candidate1i,test_batch.label_1,test_batch.mention_history,args)
                #print("sel_grp_index:",grp_index_2)
                if grp_index_2.size()[0]!=0:
                    step2,_,_=model.inference_redial(intent_index_2,tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_index_2,batch_index_2,grp_index_2,grp_bat_index_2,last_index_2,score_mask_2,word_index,word_batch_index,last_weights=last_weight1i,layer=1)
                    selected_2=select_layer_2(step2,grp_index_2.cpu().numpy(),grp_bat_index_2.cpu().numpy(),intent_index_2.cpu().numpy(),node_candidate2,cur_batch_size,args)
                else:
                    selected_2=[[[]]]

            
            
            for i in range(len(selected_1)):
                for j in range(len(selected_1[i])):
                    selected_1[i][j]=int(selected_1[i][j])
            for i in range(len(selected_2)):
                for j in range(len(selected_2[i])):
                    for k in range(len(selected_2[i][j])):
                        selected_2[i][j][k]=int(selected_2[i][j][k])
            all_intent=['chat','question','recommend']

            dataset=ReDial(args['data_path'],flag="test")

            for num in range(cur_batch_size):
                itt=intent_label[num] if golden_intent else selected[num]
                data={'intent':all_intent[itt],'layer1':selected_1[num],'layer2':selected_2[num],'key':test_batch.my_id[num]}
                DA=da_tree_serial(data,args['id2name'])
                if len(dataset[cnt].dialog_history)!=0:
                    context=dataset[cnt].dialog_history[-1]
                else:
                    context="hello"
                context=utter_lexical_redial(context,args['mid2name'])
                gpt_in=context+" @ "+DA+" &"
               
                all_gpt_in.append(gpt_in)
                generated=gener.generate(gpt_in.lower())
                cur_turn={"generated":generated,"label":utter_lexical_redial(dataset[cnt].oracle_response,args['mid2name'])}
                generated_DAs.append(data)
                generated_utters.append(cur_turn)
                cnt+=1
    
    lines = [item['generated'].strip() for item in generated_utters]
    bleu_array = []
    f1_array = []
    
    k=0
    for item in generated_utters:
        k+=1
        ground, generate = [item['label']], item['generated']
        bleu_array.append(bleu(generate, ground))
        f1_array.append(f1_score(generate, ground))

    Bleu=np.mean(bleu_array)
    f1=np.mean(f1_array)
    dist=[]
    print("BLEU:",Bleu)
    print("F1:",f1)


    tokenized = [line.split() for line in lines]
    for n in range(1, 6):
        cnt, percent = distinct_n_grams(tokenized, n)
        dist.append(percent)
        print(f'Distinct {n}-grams (cnt, percentage) = ({cnt}, {percent:.3f})')
    
    return Bleu, f1, dist


def evaluate_gen_gorecdial(test_loader:DataLoader, model:ProRec, graph_data, bow_data, args, golden_intent=True):
    model.eval()
    generated_DAs=[]
    generated_utters=[]
    all_gpt_in=[]
    cnt=0
    from generator import Generator
    gener=Generator(args['gen_conf'])
    with torch.no_grad():
        for test_batch in tqdm(test_loader):
            tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_indices,sel_batch_indices,sel_group_indices,sel_group_indices,grp_batch_indices,last_indices,intent_indices,bow_embed,rec_index,rec_batch_index,rec_golden,intent_label,label_1,label_2,score_masks,word_index,word_batch_index=model.prepare_data_gorecdial(test_batch.dialog_history,test_batch.mention_history,test_batch.intent,test_batch.node_candidate1,test_batch.node_candidate2,graph_data.edge_type,graph_data.edge_index,test_batch.label_c,test_batch.label_2,args['attribute_dict'],test_batch.rec_cand,bow_data.bow_embed)

            intent=model.get_intent(tokenized_dialog,all_length,maxlen,init_hidden)
            selected=intent.max(dim=-1).indices
            cur_batch_size=intent.size()[0]
            
            
            if golden_intent:
                step1,last_weight1,_,_=model.inference_gorecdial(intent_indices[0],tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_indices[0],sel_batch_indices[0],sel_group_indices[0],grp_batch_indices[0],last_indices[0],score_masks,word_index,word_batch_index,bow_embed,rec_index,rec_batch_index)

                step1=step1.cpu().numpy()
                step_grp=sel_group_indices[0].cpu().numpy()
                
                sel_index_2,grp_index_2,batch_index_2,intent_index_2,grp_bat_index_2,last_index_2,score_mask_2,node_candidate2,selected_1,all_split=select_layer_1(args['nodes'],step1,step_grp,intent_label,test_batch.node_candidate1,test_batch.label_1,test_batch.mention_history,args,test_batch.rec_cand,dataset="gorecdial")
                step2,_,_=model.inference_gorecdial(intent_index_2,tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_index_2,batch_index_2,grp_index_2,grp_bat_index_2,last_index_2,score_mask_2,word_index,word_batch_index,last_weights=last_weight1,layer=1)

                selected_2=select_layer_2(step2,grp_index_2.cpu().numpy(),grp_bat_index_2.cpu().numpy(),intent_index_2.cpu().numpy(),node_candidate2,cur_batch_size,args)
                
            else:
                sel_index_1i,grp_index_1i,batch_index_1i,grp_bat_index_1i,last_index_1i,node_candidate1i,score_mask1i=select_intent(selected,test_batch.mention_history,args)
                step1i,last_weight1i,_,_=model.inference_gorecdial(selected,tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_index_1i,batch_index_1i,grp_index_1i,grp_bat_index_1i,last_index_1i,score_mask1i,word_index,word_batch_index,bow_embed,rec_index,rec_batch_index)
                step_grp=grp_index_1i.cpu().numpy()

                sel_index_2,grp_index_2,batch_index_2,intent_index_2,grp_bat_index_2,last_index_2,score_mask_2,node_candidate2,selected_1,all_split=select_layer_1(args['nodes'],step1i,step_grp,selected,node_candidate1i,test_batch.label_1,test_batch.mention_history,args,test_batch.rec_cand,dataset="gorecdial")

                if grp_index_2.size()[0]!=0:
                    step2,_,_=model.inference_gorecdial(intent_index_2,tokenized_dialog,all_length,maxlen,init_hidden,edge_type,edge_index,mention_index,mention_batch_index,sel_index_2,batch_index_2,grp_index_2,grp_bat_index_2,last_index_2,score_mask_2,word_index,word_batch_index,last_weights=last_weight1i,layer=1)
                    selected_2=select_layer_2(step2,grp_index_2.cpu().numpy(),grp_bat_index_2.cpu().numpy(),intent_index_2.cpu().numpy(),node_candidate2,cur_batch_size,args)
                else:
                    selected_2=[[[]]]

            
            for i in range(len(selected_1)):
                for j in range(len(selected_1[i])):
                    selected_1[i][j]=int(selected_1[i][j])
            for i in range(len(selected_2)):
                for j in range(len(selected_2[i])):
                    for k in range(len(selected_2[i][j])):
                        selected_2[i][j][k]=int(selected_2[i][j][k])
            all_intent=['chat','question','recommend']
            dataset=GoRecDial(args['data_path'],flag="test")

            for num in range(cur_batch_size):
                itt=intent_label[num] if golden_intent else selected[num]
                data={'intent':all_intent[itt],'layer1':selected_1[num],'layer2':selected_2[num],'key':test_batch.my_id[num]}
                DA=da_tree_serial(data,args['id2name'])
                if len(dataset[cnt].dialog_history)!=0:
                    context=dataset[cnt].dialog_history[-1]
                else:
                    context="hello"
                context=utter_lexical_redial(context,args['mid2name'])
                gpt_in=context+" @ "+DA+" &"
               
                all_gpt_in.append(gpt_in)
                generated=gener.generate(gpt_in.lower())
                cur_turn={"generated":generated,"label":utter_lexical_gorecdial(dataset[cnt].oracle_response,args['mid2name'])}
                print(gpt_in)
                print(generated)
                generated_DAs.append(data)
                generated_utters.append(cur_turn)
                cnt+=1
    

    lines = [item['generated'].strip() for item in generated_utters]
    bleu_array = []
    f1_array = []
    
    k=0
    for item in generated_utters:
        k+=1
        ground, generate = [item['label']], item['generated']
        bleu_array.append(bleu(generate, ground))
        f1_array.append(f1_score(generate, ground))

    Bleu=np.mean(bleu_array)
    f1=np.mean(f1_array)
    dist=[]
    print("BLEU:",Bleu)
    print("F1:",f1)


    tokenized = [line.split() for line in lines]
    for n in range(1, 6):
        cnt, percent = distinct_n_grams(tokenized, n)
        dist.append(percent)
        print(f'Distinct {n}-grams (cnt, percentage) = ({cnt}, {percent:.3f})')

    return Bleu,f1,dist


