import torch
import json
from copy import deepcopy
import os.path as osp



args={}


def preprocess():
    if args.get('dataset') == 'tgredial':
        return preprocess_tgredial()
    with open(args['graph_path'],'r') as f:
        graph=json.load(f)
    nodes=graph["nodes"]
    relations=graph["relations"]
    attribute_dict=[]
    generals_dict={}
    generals=[]
    movie_count=0
    for i,item in enumerate(nodes):
        attribute_dict.append(set())
        if item["type"]=="Movie":
            movie_count+=1
        if item["type"]=="Attr" and item["name"]!="None":
            generals_dict[item["name"]]=set()
            generals.append(i)
    for item in relations:
        if item[0]<movie_count:
            attribute_dict[item[0]].add(item[1])
        if item[0]>=movie_count:
            if item[1]<movie_count:
                attribute_dict[item[0]].add(item[1])
    args['generals']=generals
    args['generals_dict']=generals_dict
    args['attribute_dict']=attribute_dict
    args['movie_count']=movie_count
    args['nodes']=deepcopy(graph['nodes'])
    
    all_relations={}

    for item in relations:
        if item[2] not in all_relations.keys():
            all_relations[item[2]]=1
        else:
            all_relations[item[2]]=all_relations[item[2]]+1
    print(all_relations)
    total_rel=0

    for key,value in all_relations.items():
        total_rel+=value

    print(total_rel)

    # --- Precompute KG coverage caches ---
    # movie_kg_neighbors[movie_id] = set of 1-hop neighbor entity ids
    # movie_kg_relation_types[movie_id] = set of directed relation-type ids
    relations_names = ['time', 'director', 'starring', 'genre', 'subject', 'belong',
                       'timeR', 'directorR', 'starringR', 'genreR', 'subjectR', 'belongR']
    movie_kg_neighbors = {}
    movie_kg_relation_types = {}
    for item in relations:
        src, dst, rel_name = int(item[0]), int(item[1]), item[2]
        rel_id = relations_names.index(rel_name) if rel_name in relations_names else -1
        # We care about edges originating from movie nodes
        if src < movie_count:
            if src not in movie_kg_neighbors:
                movie_kg_neighbors[src] = set()
                movie_kg_relation_types[src] = set()
            movie_kg_neighbors[src].add(dst)
            if rel_id >= 0:
                movie_kg_relation_types[src].add(rel_id)
        # Also reverse edges landing on movies (entity -> movie) give neighbor info for the entity source
        # but for coverage we only look at outgoing edges from movie nodes
    # Total reachable entities = union of all 1-hop neighbors of all movies
    total_reachable = set()
    for neighbors in movie_kg_neighbors.values():
        total_reachable |= neighbors
    args['movie_kg_neighbors'] = movie_kg_neighbors
    args['movie_kg_relation_types'] = movie_kg_relation_types
    args['total_reachable_entities'] = max(len(total_reachable), 1)
    args['total_relation_types'] = len(relations_names)  # 12 directed types
    print("Coverage caches: reachable entities =", args['total_reachable_entities'],
          ", relation types =", args['total_relation_types'])




def preprocess_tgredial():
    """Build args for TG-ReDial KG ({entity, edge, n_relation}, no node types)."""
    with open(args['graph_path'], 'r', encoding='utf-8') as f:
        graph = json.load(f)
    entities = graph['entity']
    edges = graph['edge']
    n_relation = int(graph.get('n_relation', 0))

    # Movies are listed separately in movie_ids.json. Treat that set as the
    # "movie" partition for attribute / coverage caches.
    root = osp.dirname(osp.dirname(osp.abspath(__file__)))
    with open(osp.join(root, 'data', 'tgredial', 'movie_ids.json'), 'r') as f:
        movie_ids = set(int(m) for m in json.load(f))

    movie_count = len(movie_ids)
    n_ent = len(entities)
    attribute_dict = [set() for _ in range(n_ent)]
    for src, dst, _rel in edges:
        src_i = int(src); dst_i = int(dst)
        if src_i >= n_ent or dst_i >= n_ent:
            continue  # 12 stray edges in tgredial_kg.json reference unknown ids
        if src_i in movie_ids:
            attribute_dict[src_i].add(dst_i)
        elif dst_i in movie_ids:
            attribute_dict[src_i].add(dst_i)

    args['generals'] = []
    args['generals_dict'] = {}
    args['attribute_dict'] = attribute_dict
    args['movie_count'] = movie_count
    # Eval code (model/evaluation.py:select_layer_1) expects ReDial-shaped
    # dicts with `type`/`global` fields. TG-ReDial KG has only entity names,
    # so we synthesise a minimal schema: Movies vs Attr (anything else gets
    # the 'Attr' tag, which the eval loops correctly skip for general
    # category accounting).
    args['nodes'] = [
        {
            'type': 'Movie' if i in movie_ids else 'Attr',
            'name': name,
            'global': i,
        }
        for i, name in enumerate(entities)
    ]

    movie_kg_neighbors = {}
    movie_kg_relation_types = {}
    for src, dst, rel in edges:
        src_i = int(src); dst_i = int(dst); rel_i = int(rel)
        if src_i >= n_ent or dst_i >= n_ent:
            continue
        if src_i in movie_ids:
            movie_kg_neighbors.setdefault(src_i, set()).add(dst_i)
            movie_kg_relation_types.setdefault(src_i, set()).add(rel_i)

    total_reachable = set()
    for ns in movie_kg_neighbors.values():
        total_reachable |= ns

    args['movie_kg_neighbors'] = movie_kg_neighbors
    args['movie_kg_relation_types'] = movie_kg_relation_types
    args['total_reachable_entities'] = max(len(total_reachable), 1)
    args['total_relation_types'] = n_relation
    print('[tgredial] entities=', len(entities), 'movies=', movie_count,
          'edges=', len(edges), 'relations=', n_relation)


def add_generic_args(dataset='redial'):
    args['dataset']=dataset
    args['device']=torch.device('cuda:0')
    root=osp.dirname(osp.dirname(osp.abspath(__file__)))

    # global config
    top_k=0
    top_p=0.9
    max_length=50
    temperature=1.0

    # dataset-specific config
    if dataset=='redial':
        with open(osp.join(root,"data",'id2name_redial.json'), 'r') as f:
            args['id2name']=json.load(f)
        with open(osp.join(root,"data",'mid2name_redial.json'), 'r') as f:
            args['mid2name'] = json.load(f)
        args['threshold']=[[-1,-1,-1],[-1,-1,-1]] #scoring threshold of logits for different intent at depth 1&2
        args['max_leaf']=2 #max number of leaf nodes
        args['sample']=1 #reasoning width

        args['data_path']=osp.join(root,"data",'redial')
        args['graph_path']=osp.join(root,"data","redial","raw",'redial_kg.json')
        gpt_path=osp.join(root,"data","redial_gpt")
        args['gen_conf']={'gpt_path':gpt_path,'top_k':top_k,'top_p':top_p,'max_length':max_length,'temperature':temperature}


        args['DA_save_path']=osp.join(root,"saved",'redial_DA.json')
        args['utter_save_path']=osp.join(root,"saved",'redial_gen.json')
        args['none_node']=30458
        
        
    elif dataset == 'tgredial':
        # No id2name file ships with TG-ReDial in this branch; the entity
        # name list is loaded directly from the KG inside preprocess_tgredial
        # and stored in args['nodes']. Provide empty placeholders so the
        # rest of the code that does `args['id2name'].get(...)` keeps working.
        args['id2name'] = {}
        args['mid2name'] = {}
        args['threshold'] = [[-1, -1, -1], [-1, -1, -1]]
        args['max_leaf'] = 2
        args['sample'] = 1

        args['data_path'] = osp.join(root, "data", 'tgredial')
        args['graph_path'] = osp.join(root, "data", 'tgredial_kg.json')
        gpt_path = osp.join(root, "data", "tgredial_gpt")
        args['gen_conf'] = {
            'gpt_path': gpt_path,
            'top_k': top_k,
            'top_p': top_p,
            'max_length': max_length,
            'temperature': temperature,
        }
        args['DA_save_path'] = osp.join(root, "saved", 'tgredial_DA.json')
        args['utter_save_path'] = osp.join(root, "saved", 'tgredial_gen.json')
        # Reserve a sentinel "none" id one past the entity range.
        with open(osp.join(root, 'data', 'tgredial_kg.json'), 'r', encoding='utf-8') as f:
            _kg = json.load(f)
        args['none_node'] = len(_kg['entity']) - 1

    else:
        with open(osp.join(root,"data",'id2name_gorecdial.json'), 'r') as f:
            args['id2name']=json.load(f)
        with open(osp.join(root,"data",'mid2name_gorecdial.json'), 'r') as f:
            args['mid2name'] = json.load(f)
        args['threshold']=[[-1,-1,1],[-1,-1,-1]]
        args['max_leaf']=1
        args['sample']=1
        
        args['data_path']=osp.join(root,"data",'gorecdial')
        args['graph_path']=osp.join(root,"data","gorecdial","raw",'gorecdial_kg.json')

        gpt_path=osp.join(root,"data","gorecdial_gpt")
        args['gen_conf']={'gpt_path':gpt_path,'top_k':top_k,'top_p':top_p,'max_length':max_length,'temperature':temperature}

        args['DA_save_path']=osp.join(root,"saved","gorecdial_DA.json")
        args['utter_save_path']=osp.join(root,"saved",'gorecdial_gen.json')
        args['none_node']=19307
    preprocess()
