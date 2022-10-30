
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
from sklearn.utils import shuffle as skshuffle
from model.STKGRec import STKGRec
from utility.loader_KGPOI import *
import argparse
import nni
import logging
import time
from torch.nn.parameter import Parameter

logger = logging.getLogger('STKGRec')

def train_network(network,file,args, criterion = None):
    #generation session data
    candidate = data_neural.keys()
    data_train, train_idx = generate_input_history(data_neural, 'train', candidate=candidate) 
    print('check-in data loading finish')
    #generation kg data
    train_kg_dict, train_kg = kg['train_kg_dict'], kg['train_kg']
    n_user_entity, n_loc_entity=len(data['uid_list']), len(data['vid_list'])
    tim_dis_dict = ptp_dict
    print('KG data loading finish')
    print('start training ')
    max_metric = 0
    worse_round = 0
    early_stopping_round = 15

    file.write('\t rec@1,rec@5,rec@10,ndcg@1,ndcg@5,ndcg@10 \n')
    for epoch in range(args['epochs']):
        network.train(True)
        i,j = 0,0 
        total_loss, poi_loss, kg_loss=0,0,0
        run_queue = generate_queue(train_idx, 'random', 'train')  #list( [user_id, session_id])

        for one_train_batch in minibatch(run_queue, batch_size = args['batch_size'] ):
            user_id_batch, session_id_batch, sequence_batch, sequences_lens_batch, sequence_tim_batch, \
            tim_dis_gap_batch,sequences_dilated_input_batch = generate_detailed_batch_data(n_loc_entity,one_train_batch,data_neural,tim_dis_dict,n_tim_rel,poi_temporal_distance_matrix)
            max_len = max(sequences_lens_batch)
            padded_sequence_batch, tim_sequence_batch, mask_batch_ix, mask_batch_ix_non_local,\
            padded_tim_dis_gap_batch = pad_batch_of_lists_masks(sequence_batch,sequence_tim_batch, tim_dis_gap_batch, max_len) 
            padded_sequence_batch = Variable(torch.LongTensor(np.array(padded_sequence_batch))).to(device)
            tim_sequence_batch = Variable(torch.LongTensor(np.array(tim_sequence_batch))).to(device)
            padded_tim_dis_gap_batch = Variable(torch.LongTensor(np.array(padded_tim_dis_gap_batch))).to(device)
            mask_batch_ix = Variable(torch.FloatTensor(np.array(mask_batch_ix))).to(device)
            mask_batch_ix_non_local = Variable(torch.FloatTensor(np.array(mask_batch_ix_non_local))).to(device)
            user_id_batch = Variable(torch.LongTensor(np.array(user_id_batch))).to(device)
            logp_seq = network('predict',user_id_batch, padded_sequence_batch, mask_batch_ix_non_local, session_id_batch, 
            tim_sequence_batch,padded_tim_dis_gap_batch,device, True, sequences_dilated_input_batch)
            predictions_logp = logp_seq[:, :-1] * mask_batch_ix[:, :-1, None]
            actual_next_tokens = padded_sequence_batch[:, 1:]
            logp_next = torch.gather(predictions_logp, dim=2, index=actual_next_tokens[:, :, None])
            loss = -logp_next.sum() / mask_batch_ix[:, :-1].sum()

            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            opt.step()
            opt.zero_grad()
            poi_loss += loss.item()
            if (i + 1) % 80== 0:
                print("epoch" + str(epoch) + ": loss: " + str(loss))
                logger.debug("epoch" + str(epoch) + ": loss: " + str(loss))
            i += 1


        for kg_train_batch in minibatch(train_kg, batch_size = args['batch_size']):
            batch_head, batch_relation, batch_pos_tail, batch_neg_tail = generate_kg_batch(train_kg_dict,kg_train_batch,n_loc_entity,device)
            batch_loss = network('calc_kg_loss',batch_head,batch_relation,batch_pos_tail,batch_neg_tail)
            batch_loss.backward()
            opt.step()
            opt.zero_grad()
            kg_loss += batch_loss.item()
            j += 1

        print("epoch" + str(epoch),"total loss:",(poi_loss/i)+(kg_loss/j))
        # if not os.path.isdir('./checkpoint'):
        #     os.mkdir('./checkpoint')
        if epoch >15:
            metric = evaluate(network,'vaild', 1)
            if max_metric < metric[1]:
                max_metric = metric[1]
                worse_round = 0
                torch.save(network.state_dict(),'./checkpoint/'+str(params['data'])+'_'+str(params['batch_size'])+'_'+str(params['lr'])+\
                '_'+str(params['hidden_size'])+'_model.pkl')
            else:
                worse_round += 1
            if 0 < early_stopping_round <= worse_round:
                print('Early Stopping, best Epoch %d.' % (epoch - worse_round))
                break

            file.write("epoch" + str(epoch)+"\t Scores:\t"+str(metric)+'\n')
            print("epoch" + str(epoch),"Scores: ", metric)
            # nni.report_intermediate_result(metric[1])

    # Load the model with the best test metric value.
    network.load_state_dict(torch.load('./checkpoint/'+str(params['data'])+'_'+str(params['batch_size'])+'_'+str(params['lr'])+\
               '_'+str(params['hidden_size'])+'_model.pkl'))
    results = evaluate(network, 'test', 1)
    print('best result', results)
    # nni.report_intermediate_result(results[1])
    file.write("best result' Scores:\t"+str(results)+'\n')



def get_acc(target, scores):
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(10, 1)
    predx = idxx.cpu().numpy()
    acc = np.zeros((3, 1))
    ndcg = np.zeros((3, 1))
    for i, p in enumerate(predx):
        t = target[i]
        if t != 0:
            if t in p[:10] and t > 0:
                acc[0] += 1
                rank_list = list(p[:10])
                rank_index = rank_list.index(t)
                ndcg[0] += 1.0 / np.log2(rank_index + 2)
            if t in p[:5] and t > 0:
                acc[1] += 1
                rank_list = list(p[:5])
                rank_index = rank_list.index(t)
                ndcg[1] += 1.0 / np.log2(rank_index + 2)
            if t == p[0] and t > 0:
                acc[2] += 1
                rank_list = list(p[:1])
                rank_index = rank_list.index(t)
                ndcg[2] += 1.0 / np.log2(rank_index + 2)
        else:
            break
    return acc.tolist(), ndcg.tolist()

def evaluate(network,mode, batch_size =64):
    network.train(False)
    candidate = data_neural.keys()
    n_user_entity, n_loc_entity=len(data['uid_list']), len(data['vid_list'])
    data_test, test_idx = generate_input_long_history(data_neural, mode, candidate=candidate)
    users_acc = {}
    with torch.no_grad():
        tim_dis_dict = ptp_dict
        run_queue = generate_queue(test_idx, 'normal', mode)
        for one_test_batch in minibatch(run_queue, batch_size=batch_size):
            user_id_batch, session_id_batch, sequence_batch, sequences_lens_batch, sequence_tim_batch,\
            tim_dis_gap_batch,sequences_dilated_input_batch = generate_detailed_batch_data(n_loc_entity,one_test_batch,data_neural,tim_dis_dict,n_tim_rel,poi_temporal_distance_matrix)
            user_id_batch_test = user_id_batch
            max_len = max(sequences_lens_batch)
            padded_sequence_batch, tim_sequence_batch, mask_batch_ix, mask_batch_ix_non_local,\
            padded_tim_dis_gap_batch = pad_batch_of_lists_masks(sequence_batch,sequence_tim_batch,tim_dis_gap_batch,max_len)
            padded_sequence_batch = Variable(torch.LongTensor(np.array(padded_sequence_batch))).to(device)
            tim_sequence_batch = Variable(torch.LongTensor(np.array(tim_sequence_batch))).to(device)
            padded_tim_dis_gap_batch = Variable(torch.LongTensor(np.array(padded_tim_dis_gap_batch))).to(device)
            mask_batch_ix = Variable(torch.FloatTensor(np.array(mask_batch_ix))).to(device)
            mask_batch_ix_non_local = Variable(torch.FloatTensor(np.array(mask_batch_ix_non_local))).to(device)
            user_id_batch = Variable(torch.LongTensor(np.array(user_id_batch))).to(device)
            logp_seq = network('predict',user_id_batch, padded_sequence_batch, mask_batch_ix_non_local, session_id_batch,\
            tim_sequence_batch,padded_tim_dis_gap_batch,device, False,sequences_dilated_input_batch)
            predictions_logp = logp_seq[:, :-1] * mask_batch_ix[:, :-1, None]
            actual_next_tokens = padded_sequence_batch[:, 1:]
            for ii, u_current in enumerate(user_id_batch_test):
                if u_current not in users_acc:
                    users_acc[u_current] = [0, 0, 0, 0, 0, 0, 0]
                acc, ndcg = get_acc(actual_next_tokens[ii], predictions_logp[ii])
                users_acc[u_current][1] += acc[2][0]#@1
                users_acc[u_current][2] += acc[1][0]#@5
                users_acc[u_current][3] += acc[0][0]#@10
                ###ndcg
                users_acc[u_current][4] += ndcg[2][0]  # @1
                users_acc[u_current][5] += ndcg[1][0]  # @5
                users_acc[u_current][6] += ndcg[0][0]  # @10
                users_acc[u_current][0] += (sequences_lens_batch[ii]-1)
        tmp_acc = [0.0,0.0,0.0, 0.0, 0.0, 0.0]##last 3 ndcg
        sum_test_samples = 0.0
        for u in users_acc:
            tmp_acc[0] = users_acc[u][1] + tmp_acc[0]
            tmp_acc[1] = users_acc[u][2] + tmp_acc[1]
            tmp_acc[2] = users_acc[u][3] + tmp_acc[2]

            tmp_acc[3] = users_acc[u][4] + tmp_acc[3]
            tmp_acc[4] = users_acc[u][5] + tmp_acc[4]
            tmp_acc[5] = users_acc[u][6] + tmp_acc[5]
            sum_test_samples = sum_test_samples + users_acc[u][0]
        avg_acc = (np.array(tmp_acc)/sum_test_samples).tolist()
        return avg_acc

def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='STKGRec')
    parser.add_argument("--data_dir", type=str,
                        default="./data/",help="data directory")
    parser.add_argument("--data", type=str,
                        default="nyc",help="data name")
    parser.add_argument("--gpu", type=int,
                        default="1",help="choose gpu")
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument("--hidden_size", type=int, default=100, metavar='N',
                        help='hidden layer size')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--wd', type=float, default=1 * 1e-6, metavar='LR',
                        help='weight_decay')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--kg_lambda', type=float, default=1e-5,
                        help='Lambda when calculating KG l2 loss.')
    parser.add_argument('--td', type=float, default=0.4,
                        help='Lambda between temporal and distance.')
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
    args = get_params()
    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    params = vars(get_params())
    params.update(tuner_params)
    print(params)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(params['gpu'])
    path = params['data_dir'] +params['data'] + '.pkl'
    data = pickle.load(open(path, 'rb'), encoding='iso-8859-1')

    vid_list = data['vid_list']
    uid_list = data['uid_list']
    kg = data['KG']
    data_neural = data['data_neural'] 
    poi_coordinate = data['vid_lookup']   
    tim_gap = kg['tim_rel']
    dis_gap = kg['dis_rel']
    poi_trans = kg['poi_trans']
    ptp_dict = kg['ptp_dict']   
    tim_max,dis_max,tim_rel=kg['max_dis_tim'][0],kg['max_dis_tim'][1],kg['timining_rel']
    if os.path.exists(params['data_dir'] +params['data']+'_temporal_distance.pkl'):
        poi_temporal_distance_matrix = pickle.load(open(params['data_dir'] +params['data'] +'_temporal_distance.pkl', 'rb'),encoding='iso-8859-1')
        print("load temporal dictance data! ")
    else:
        poi_temporal_distance_matrix = caculate_poi_distance_time(params,poi_coordinate,poi_trans,tim_max,dis_max)
        print("file find!")
    loc_size = len(vid_list)
    uid_size = len(uid_list)
    ptp_size = len(ptp_dict.keys())
    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device("cuda")
    n_users = uid_size
    n_locs = loc_size
    n_tim_gap = len(tim_gap)
    n_dis_gap = len(dis_gap)
    n_tim_rel = len(tim_rel)
    print('user and loc num:',n_users,n_locs)
    session_id_sequences = None
    user_id_session = None
    network = STKGRec(params,n_users=n_users, n_locs=n_locs,n_tim_rel=n_tim_rel, n_tim_dis_gap=ptp_size,\
    data_neural=data_neural, kg=kg).to(device)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr=params['lr'],\
                               weight_decay=params['wd'])
    criterion = nn.NLLLoss().cuda()
    ticks = int(time.time())
    if not os.path.isdir('./result'):
        os.mkdir('./result')
    result_path = open('./result/'+str(ticks) +str(params['data']) +'_'+ str(params['batch_size']) +'_' +str(params['lr'])+\
               '_'+str(params['hidden_size'])+'.txt','a')
    result_path.write(str(params)+'\n')
    train_network(network,result_path,params, criterion=criterion)
