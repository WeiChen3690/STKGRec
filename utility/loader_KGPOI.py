
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pickle
import numpy as np
import torch
from collections import defaultdict
import gc
import os
from math import radians, cos, sin, asin, sqrt
from collections import deque,Counter
import collections
from sklearn.utils import shuffle as skshuffle
from tqdm import tqdm

from torch.nn.parameter import Parameter

def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)
    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')
    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)
    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)
    if require_indices:
        return result, shuffle_indices
    else:
        return result

def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', 32)
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

def get_minibatches(X, mb_size, shuffle=True):
    X_shuff = X.copy()
    if shuffle:
        X_shuff = skshuffle(X_shuff)

    for i in range(0, X_shuff.shape[0], mb_size):
        yield X_shuff[i:i + mb_size]

def pad_batch_of_lists_masks(batch_of_lists, sequence_tim_batch, tim_dis_gap_batch, max_len):
    loc_padded = [l + [0] * (max_len - len(l)) for l in batch_of_lists] 
    tim_padded = [l + [0] * (max_len - len(l)) for l in sequence_tim_batch]
    tim_dis_gap_padded = [l + [0] * (max_len - len(l)) for l in tim_dis_gap_batch]
    # dis_gap_padded = [l + [0] * (max_len - len(l)) for l in dis_gap_batch]
    padded_mask = [[1.0]*(len(l) - 1) + [0.0] * (max_len - len(l) + 1) for l in batch_of_lists] 
    padde_mask_non_local = [[1.0] * (len(l)) + [0.0] * (max_len - len(l)) for l in batch_of_lists] 
    return loc_padded, tim_padded, padded_mask, padde_mask_non_local, tim_dis_gap_padded

def pad_batch_of_lists_masks_test(batch_of_lists, max_len):
    padded = [l + [0] * (max_len - len(l)) for l in batch_of_lists]
    padded2 = [l[:-1] + [0] * (max_len - len(l) + 1) for l in batch_of_lists]
    padded_mask = [[0.0]*(len(l) - 2) + [1.0] + [0.0] * (max_len - len(l) + 1) for l in batch_of_lists]
    padde_mask_non_local = [[1.0] * (len(l) - 1) + [0.0] * (max_len - len(l) + 1) for l in batch_of_lists]
    return padded, padded2, padded_mask, padde_mask_non_local

def caculate_poi_distance_time(args, poi_coors, trans, temp_max, distance_max):
    print("temporal distance matrix")
    tem_dis_score = {}
    for poi_gap in trans:
        time_between = 0
        for id in trans[poi_gap]:
            time_between +=  id[0]
        tem_dis_score[poi_gap]= ((time_between/len(trans[poi_gap]))/temp_max)* args['td'] + (id[1]/distance_max)*(1-args["td"])
    sim_matrix = np.zeros((len(poi_coors) + 1, len(poi_coors) + 1))
    for i in tqdm(range(len(poi_coors))):
        for j in range(i , len(poi_coors)):
            poi_current = i + 1
            poi_target = j + 1
            poi_current_coor = poi_coors[poi_current]
            poi_target_coor = poi_coors[poi_target]
            if tuple([i,j]) in tem_dis_score:
                sim_matrix[poi_current][poi_target] = tem_dis_score[tuple([i,j])]
                sim_matrix[poi_target][poi_current] = tem_dis_score[tuple([i,j])]
            distance_between = geodistance(poi_current_coor[1], poi_current_coor[0], poi_target_coor[1], poi_target_coor[0])
            distance_between = int(np.float(distance_between)/0.06)
            sim_matrix[poi_current][poi_target] = distance_between/distance_max
            sim_matrix[poi_target][poi_current] = distance_between/distance_max
    # pickle.dump(sim_matrix, open(args['data_dir'] + args['data'] +'_temporal_distance.pkl', 'wb'))
    return sim_matrix


def generate_input_history(data_neural, mode, candidate=None):
    data_train = {}
    train_idx = {}  
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:  
        sessions = data_neural[u]['sessions'] 
        train_id = data_neural[u][mode]    
        data_train[u] = {}     

        for c, i in enumerate(train_id):   
            if mode == 'train' and c == 0:
                continue
            session = sessions[i]   
            trace = {} 

            loc_np = np.reshape(np.array([s[0] for s in session[:-1]]), (len(session[:-1]), 1))
            tim_np = np.reshape(np.array([s[1] for s in session[:-1]]), (len(session[:-1]), 1))
            target = np.array([s[0] for s in session[1:]])   
            trace['loc'] = Variable(torch.LongTensor(loc_np))  
            trace['target'] = Variable(torch.LongTensor(target))
            trace['tim'] = Variable(torch.LongTensor(tim_np))
            history = []
            if mode == 'test' or mode == 'vaild':
                test_id = data_neural[u]['train']
                for tt in test_id:
                    history.extend([(s[0], s[1]) for s in sessions[tt]])
            for j in range(c):
                history.extend([(s[0], s[1]) for s in sessions[train_id[j]]])
            history = sorted(history, key=lambda x: x[1], reverse=False)  
            history_loc = np.reshape(np.array([s[0] for s in history]), (len(history), 1)) 
            history_tim = np.reshape(np.array([s[1] for s in history]), (len(history), 1))
            trace['history_loc'] = Variable(torch.LongTensor(history_loc))
            trace['history_tim'] = Variable(torch.LongTensor(history_tim))
            data_train[u][i] = trace 
        train_idx[u] = train_id 
    return data_train, train_idx

def generate_input_long_history(data_neural, mode, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}
        for c, i in enumerate(train_id):
            trace = {}
            if mode == 'train' and c == 0:
                continue
            session = sessions[i]
            target = np.array([s[0] for s in session[1:]])
            history = []
            if mode == 'test' or mode == 'vaild':
                test_id = data_neural[u]['train']
                for tt in test_id:
                    history.extend([(s[0], s[1]) for s in sessions[tt]])
            for j in range(c):
                history.extend([(s[0], s[1]) for s in sessions[train_id[j]]])
            history_tim = [t[1] for t in history]
            history_count = [1]
            last_t = history_tim[0]
            count = 1
            for t in history_tim[1:]:
                if t == last_t:
                    count += 1
                else:
                    history_count[-1] = count
                    history_count.append(1)
                    last_t = t
                    count = 1
            history_loc = np.reshape(np.array([s[0] for s in history]), (len(history), 1))
            history_tim = np.reshape(np.array([s[1] for s in history]), (len(history), 1))
            trace['history_loc'] = Variable(torch.LongTensor(history_loc))
            trace['history_tim'] = Variable(torch.LongTensor(history_tim))
            trace['history_count'] = history_count
            loc_tim = history
            loc_tim.extend([(s[0], s[1]) for s in session[:-1]])
            loc_np = np.reshape(np.array([s[0] for s in loc_tim]), (len(loc_tim), 1))
            tim_np = np.reshape(np.array([s[1] for s in loc_tim]), (len(loc_tim), 1))
            trace['loc'] = Variable(torch.LongTensor(loc_np))
            trace['tim'] = Variable(torch.LongTensor(tim_np))
            trace['target'] = Variable(torch.LongTensor(target))
            data_train[u][i] = trace
        train_idx[u] = train_id
    return data_train, train_idx

def generate_queue(train_idx, mode, mode2):
    user = list(train_idx.keys())
    train_queue = list()
    if mode == 'random':
        initial_queue = {}
        for u in user:
            if mode2 == 'train':
                initial_queue[u] = deque(train_idx[u][1:])  
            else:
                initial_queue[u] = deque(train_idx[u])
        queue_left = 1
        while queue_left > 0:
            for j, u in enumerate(user):
                if len(initial_queue[u]) > 0:
                    train_queue.append((u, initial_queue[u].popleft()))   
            queue_left = sum([1 for x in initial_queue if len(initial_queue[x]) > 0])
    elif mode == 'normal':
        for u in user:
            for i in train_idx[u]:
                train_queue.append((u, i))
    return train_queue


def create_dilated_rnn_input(session_sequence_current,poi_temporal_distance_matrix):
    sequence_length = len(session_sequence_current)
    session_sequence_current.reverse()
    session_dilated_rnn_input_index = [0] * sequence_length
    for i in range(sequence_length - 1):
        current_poi = [session_sequence_current[i]]
        poi_before = session_sequence_current[i + 1 :]
        temporal_distance_row = poi_temporal_distance_matrix[current_poi]
        temporal_distance_row_explicit = temporal_distance_row[:, poi_before][0]
        index_closet = np.argmin(temporal_distance_row_explicit)
        session_dilated_rnn_input_index[sequence_length - i - 1] = sequence_length-2-index_closet-i
    session_sequence_current.reverse()
    return session_dilated_rnn_input_index


def generate_detailed_batch_data(n_locs,one_train_batch,data_neural,tim_dis_dict,n_tim_rel,poi_temporal_distance_matrix):
    session_id_batch = []
    user_id_batch = []
    sequence_batch = []
    sequences_lens_batch = []
    sequences_tim_batch = []
    tim_dis_gap_batch =[]
    sequences_dilated_input_batch = []
    for sample in one_train_batch:
        user_id_batch.append(int(sample[0]+n_locs))  
        session_id_batch.append(sample[1]) 
        session_sequence_current = [s[0] for s in data_neural[sample[0]]['sessions'][sample[1]]] 
        session_sequence_tim_current = [s[1] for s in data_neural[sample[0]]['sessions'][sample[1]]]
        session_sequence_tim_dis = [int(tim_dis_dict[tuple(s[1])]+n_tim_rel) for s in data_neural[sample[0]]['sessions_trans'][sample[1]]]
        session_sequence_dilated_input = create_dilated_rnn_input(session_sequence_current,poi_temporal_distance_matrix) 
        sequence_batch.append(session_sequence_current) 
        sequences_lens_batch.append(len(session_sequence_current))
        sequences_tim_batch.append(session_sequence_tim_current)
        tim_dis_gap_batch.append(session_sequence_tim_dis)

        sequences_dilated_input_batch.append(session_sequence_dilated_input)
    return user_id_batch, session_id_batch, sequence_batch, sequences_lens_batch, sequences_tim_batch,tim_dis_gap_batch,sequences_dilated_input_batch


def construct_data(kg_upt_data,kg_ptp_data,tim_dis_dict,n_tim_rel):
    train_kg_ptp,train_kg_upt, train_kg =[],[],[]
    train_kg_dict = collections.defaultdict(list)
    # get utp-triple 
    head_upt = [triple[0] for triple in kg_upt_data]
    rel_upt = [triple[1] for triple in kg_upt_data]
    tail_upt = [triple[2] for triple in kg_upt_data]
    # get ptp-triple
    head_ptp = [triple[0] for triple in kg_ptp_data]
    rel_ptp = [int(tim_dis_dict[tuple(triple[1])]+n_tim_rel) for triple in kg_ptp_data]
    tail_ptp = [triple[2] for triple in kg_ptp_data]
    
    for i in range(len(head_upt)):
        if [head_upt[i], rel_upt[i],tail_upt[i]] not in train_kg:
            train_kg_dict[head_upt[i]].append((tail_upt[i], rel_upt[i]))
            train_kg.append([head_upt[i],rel_upt[i],tail_upt[i]])
    for j in range(len(head_ptp)):
        if [head_ptp[j],rel_ptp[j],tail_ptp[j]] not in train_kg:
            train_kg_dict[head_ptp[j]].append((tail_ptp[j], rel_ptp[j]))
            train_kg.append([head_ptp[j],rel_ptp[j],tail_ptp[j]])
    print('load KG data.')
    return train_kg_dict, train_kg

def generate_kg_batch(kg_dict,train_batch,n_entity,device):
    
    batch_head,batch_relation, batch_pos_tail, batch_neg_tail = [], [], [], []
    for triple in train_batch:
        head, relation, pos_tail = triple[0],triple[1],triple[2]
        batch_head.append(head)
        batch_relation.append(relation)
        batch_pos_tail.append(pos_tail)

        neg_tail = sample_neg_triples_for_h(kg_dict,n_entity, head, relation, 1)
        batch_neg_tail += neg_tail

    batch_head =Variable(torch.LongTensor(np.array(batch_head))).to(device)
    batch_relation = Variable(torch.LongTensor(np.array(batch_relation))).to(device)
    batch_pos_tail = Variable(torch.LongTensor(np.array(batch_pos_tail))).to(device)
    batch_neg_tail = Variable(torch.LongTensor(np.array(batch_neg_tail))).to(device)
    return batch_head, batch_relation, batch_pos_tail, batch_neg_tail

def sample_neg_triples_for_h(kg_dict,n_entities, head, relation, n_sample_neg_triples):
    pos_triples = kg_dict[head]

    sample_neg_tails = []
    while True:
        if len(sample_neg_tails) == n_sample_neg_triples:
            break
        tail = np.random.randint(low=0, high=n_entities, size=1)[0]
        if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
            sample_neg_tails.append(tail)
    return sample_neg_tails

def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000
    distance=round(distance/1000,3)
    return distance
