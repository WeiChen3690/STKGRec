from __future__ import print_function
from __future__ import division

import time
import datetime
import argparse
import numpy as np
import pickle as pickle
from collections import Counter
import collections
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm

from numpy.lib.function_base import select


class DataFoursquare(object):
    def __init__(self, trace_min=10, global_visit=10, hour_gap=48, min_gap=10, session_min=3, session_max=10,
                 sessions_min=3, train_split=0.8, time_split=3600, distance_split=0.06 ):
        tmp_path = "data/"
        self.TWITTER_PATH = tmp_path + 'dataset_TSMC2014_NYC.txt'
        self.SAVE_PATH = tmp_path
        self.save_name = '/nyc'

        self.trace_len_min = trace_min
        self.location_global_visit_min = global_visit
        self.hour_gap = hour_gap
        self.min_gap = min_gap
        self.session_max = session_max
        self.filter_short_session = session_min
        self.sessions_count_min = sessions_min
        self.time_split = time_split
        self.distance_split = distance_split

        self.train_split = train_split

        self.data = {}
        self.venues = {}
        self.data_filter = {}
        self.user_filter3 = None
        self.uid_list = {}
        self.vid_list = {'unk': [0, -1]}
        self.vid_list_lookup = {}
        self.vid_lookup = {}
        self.tim_gap_list = {}
        self.dis_gap_list = {}
        self.pid_loc_lat = {}
        self.data_neural = {}
        self.data_temp ={}
        self.temp_dis_dict = {}
        self.kg = {'utp':{},'ptp':{}}
        self.triple_utp,self.triple_ptp = [],[]
        self.tim,self.tim_rel,self.dis_rel,self.tim_dis_rel = [],[],[],[]
        self.train_utp,self.train_ptp = [],[]
        self.test_utp, self.test_ptp = [], []
        self.check_in_num = 0
        self.sessions_num = 0
        self.tim_max = 0
        self.dis_max = 0

    # ############# 1. read trajectory data from file
    def load_trajectory_from_tweets(self):
        with open(self.TWITTER_PATH,'rb') as fid:
            for line in fid:
                line = line.decode("utf8","ignore")
                uid, pid, _, _, lat, lon, _, UTC_time = line.strip('\n').split('\t')
                time = datetime.datetime.strptime(UTC_time,"%a %b %d %H:%M:%S %z %Y") 
                tim = time.strftime("%Y-%m-%d %H:%M:%S") 
                if uid not in self.data:      #[u1]={[loc1, time1],[loc2,time2]}
                    self.data[uid] = [[pid, tim]]
                else:
                    self.data[uid].append([pid, tim])
                if pid not in self.venues:  
                    self.venues[pid] = 1
                else:
                    self.venues[pid] += 1
                
                # read lon and lat
                self.pid_loc_lat[pid] = [float(lon), float(lat)]


    # ########### 2.0 basically filter users based on visit length and other statistics
    def filter_users_by_length(self): # self.data:[uid]={[loc,time]}
        uid_3 = [x for x in self.data if len(self.data[x]) >= self.trace_len_min]   
        pick3 = sorted([(x, len(self.data[x])) for x in uid_3], key=lambda x: x[1], reverse=True)  
        pid_3 = [x for x in self.venues if self.venues[x] >= self.location_global_visit_min]  
        pid_pic3 = sorted([(x, self.venues[x]) for x in pid_3], key=lambda x: x[1], reverse=True)   
        pid_3 = dict(pid_pic3)  

        session_len_list = []
        for u in pick3: 
            uid = u[0]
            info = self.data[uid]
            topk = Counter([x[0] for x in info]).most_common()  
            topk1 = [x[0] for x in topk if x[1] >1] 
            sessions = {}
            for i, record in enumerate(info):  
                poi, tmd = record
                try:
                    tid = int(time.mktime(time.strptime(tmd, "%Y-%m-%d %H:%M:%S"))) 
                except Exception as e:
                    print('error:{}'.format(e))
                    continue
                sid = len(sessions)
                if poi not in pid_3 and poi not in topk1: 
                    # if poi not in topk1:
                    continue
                if i == 0 or len(sessions) == 0: #session[user]={[loc1,tim1],[loc2,tim2]}
                    sessions[sid] = [record]
                else:
                    if (tid - last_tid) / 3600 > self.hour_gap or len(sessions[sid - 1]) > self.session_max: 
                        sessions[sid] = [record]
                    elif (tid - last_tid) / 60 > self.min_gap: 
                        sessions[sid - 1].append(record)
                    else:
                        pass
                last_tid = tid
            sessions_filter = {}
            for s in sessions:
                if len(sessions[s]) >= self.filter_short_session:
                    sessions_filter[len(sessions_filter)] = sessions[s]
                    session_len_list.append(len(sessions[s])) 
            if len(sessions_filter) >= self.sessions_count_min: 
                self.data_filter[uid] = {'sessions_count': len(sessions_filter), 'topk_count': len(topk), 'topk': topk,
                                         'sessions': sessions_filter, 'raw_sessions': sessions}
            # print(self.data_filter[uid])


        self.user_filter3 = [x for x in self.data_filter if
                             self.data_filter[x]['sessions_count'] >= self.sessions_count_min]

    # ########### 3. build dictionary for users and location
    def build_users_locations_dict(self):   #loc->loc_id
        for u in self.user_filter3: 
            sessions = self.data_filter[u]['sessions']
            if u not in self.uid_list:
                self.uid_list[u] = [len(self.uid_list), len(sessions)] 
            for sid in sessions:  
                poi = [p[0] for p in sessions[sid]]  
                for p in poi:
                    if p not in self.vid_list:
                        self.vid_list_lookup[len(self.vid_list)] = p #loc->loc_id
                        self.vid_list[p] = [len(self.vid_list), 1] 
                    else:
                        self.vid_list[p][1] += 1

    # ########### 4. remap lon and lat
    def venues_lookup(self):  
        for vid in self.vid_list_lookup:
            pid = self.vid_list_lookup[vid]
            lon_lat = self.pid_loc_lat[pid]
            self.vid_lookup[vid] = lon_lat

    # ########## 5.0 prepare training data for neural network
    @staticmethod
    def tid_list(tmd):
        tm = time.strptime(tmd, "%Y-%m-%d %H:%M:%S")
        tid = tm.tm_wday * 24 + tm.tm_hour
        return tid

    @staticmethod
    def tid_list_48(tmd):
        tm = time.strptime(tmd, "%Y-%m-%d %H:%M:%S")
        if tm.tm_wday in [0, 1, 2, 3, 4]:
            tid = tm.tm_hour
        else:
            tid = tm.tm_hour + 24
        return tid

    @staticmethod
    def distance(lng1,lat1,lng2,lat2):
        lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
        dlon=lng2-lng1
        dlat=lat2-lat1
        a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        distance=2*asin(sqrt(a))*6371*1000
        distance=round(distance/1000,3)
        return distance

    @staticmethod
    def ptp_dict(tim_dis_rel):
        tim_dis = {}
        count = 0
        for id in tim_dis_rel:
            if tuple(id) not in tim_dis.keys():
                tim_dis[tuple(id)]=count
                count =count+1
        return tim_dis

    @staticmethod
    def filtering_triple(triple):
        triple1 = []
        count =0
        for tri in triple:
            count+=1
            print(count)
            if tri not in triple1:
                triple1.append(tri)
        return triple1




    def prepare_neural_data(self):
        num =0
        for u in tqdm(self.uid_list):
            uid = self.uid_list[u][0]
            sessions = self.data_filter[u]['sessions']
            sessions_check = {}
            sessions_trans ={}
            sessions_id = []
            sessions_utp = {}
            self.sessions_num += len(sessions)
            num +=1
            for sid in sessions:  
                trans = []
                gap_list = []
                for i in range(len(sessions[sid])-1):   
                    # build triple (user, tim, loc)
                    j = i + 1
                    self.check_in_num += 1

                    pid = self.vid_list[sessions[sid][i][0]][0]
                    ti_pre = int(time.mktime(time.strptime(sessions[sid][i][1], "%Y-%m-%d %H:%M:%S")))
                    lat_pre = float(self.vid_lookup[pid][0])
                    lon_pre= float(self.vid_lookup[pid][1])
                    
                    pid_next = self.vid_list[sessions[sid][j][0]][0]
                    ti_next = int(time.mktime(time.strptime(sessions[sid][j][1], "%Y-%m-%d %H:%M:%S")))
                    lat_next = float(self.vid_lookup[pid_next][0])
                    lon_next = float(self.vid_lookup[pid_next][1])  

                    t_gap = ti_next-ti_pre
                    d_gap = self.distance(lon_pre,lat_pre,lon_next,lat_next)
                    t = int(np.float(t_gap)/self.time_split)
                    d = int(np.float(d_gap)/self.distance_split)
                    self.tim_max = max(self.tim_max, t)
                    self.dis_max = max(self.dis_max, d)
                    gap_list.append([t,d])
                    pid_gap = tuple([pid,pid_next])
                    if pid_gap not in self.temp_dis_dict:
                        self.temp_dis_dict[pid_gap]=[[t,d]]
                    else:
                        self.temp_dis_dict[pid_gap].append([t,d])
                    trans.append([pid,tuple([t,d]), pid_next])  
                #保存所需的数据，check-in data， transfer data
                sessions_check[sid] = [[self.vid_list[p[0]][0], self.tid_list_48(p[1])] for p in
                                      sessions[sid]]  
                sessions_utp[sid] = [[uid, self.tid_list_48(p[1]), self.vid_list[p[0]][0]] for p in
                                      sessions[sid]]  
                t_list = [self.tid_list_48(p[1]) for p in sessions[sid]]
                self.tim.extend(t_list)
                self.tim_rel.extend([ k[0] for k in gap_list])
                self.dis_rel.extend([ k[1] for k in gap_list])
                self.tim_dis_rel.extend(gap_list)
                self.triple_utp.extend(sessions_utp[sid])
                self.triple_ptp.extend(trans)
                sessions_trans[sid] = trans
                sessions_id.append(sid) 

            split_train_id = int(np.floor(self.train_split * len(sessions_id)))
            split_vaild_id = int(np.float(0.9 * len(sessions_id)))
            train_id = sessions_id[:split_train_id] 
            vaild_id = sessions_id[split_train_id:split_vaild_id]
            test_id = sessions_id[split_vaild_id:]  
            self.data_neural[self.uid_list[u][0]] = {'sessions': sessions_check, 'train': train_id, 'test': test_id,'vaild':vaild_id,
                                                     'sessions_trans':sessions_trans}
            self.data_temp[self.uid_list[u][0]] = {'sessions_utp': sessions_utp,'sessions_id': sessions_id }


    def construct_data(self,n_tim_rel,tim_dis_dict):
        train_kg_ptp,train_kg_upt, train_kg =[],[],[]
        train_kg_dict, train_kg = collections.defaultdict(list),collections.defaultdict(list)
        n_locs = len(self.vid_list)
        # get utp-triple 
        head_upt = [(triple[0]+n_locs) for triple in self.train_utp]
        rel_upt = [triple[1] for triple in self.train_utp]
        tail_upt = [triple[2] for triple in self.train_utp]
        # get ptp-triple
        head_ptp = [triple[0] for triple in self.train_ptp]
        rel_ptp = [int(tim_dis_dict[tuple(triple[1])]+n_tim_rel) for triple in self.train_ptp]
        tail_ptp = [triple[2] for triple in self.train_ptp]
        print("---------start utp------------")
        for i in tqdm(range(len(head_upt))):
            if [head_upt[i], rel_upt[i],tail_upt[i]] not in train_kg['utp']:
                train_kg_dict[head_upt[i]].append((tail_upt[i], rel_upt[i]))
                train_kg['utp'].append([head_upt[i],rel_upt[i],tail_upt[i]])
        print("---------start ptp------------")
        for j in tqdm(range(len(head_ptp))):
            if [head_ptp[j],rel_ptp[j],tail_ptp[j]] not in train_kg['ptp']:
                train_kg_dict[head_ptp[j]].append((tail_ptp[j], rel_ptp[j]))
                train_kg['ptp'].append([head_ptp[j],rel_ptp[j],tail_ptp[j]])
        print('load KG data.')
        return train_kg_dict, train_kg


    def prepare_kg_data(self):
        for u in self.data_neural:
            for k in self.data_neural[u]['train']:
                self.train_utp.extend([ p for p in self.data_temp[u]['sessions_utp'][k]])
                self.train_ptp.extend([ q for q in  self.data_neural[u]['sessions_trans'][k]])
        utp_triple, ptp_triple =self.triple_utp,self.triple_ptp
        n_tim_rel = len(list(set(self.tim)))
        tim_dis_dict = self.ptp_dict(self.tim_dis_rel)
        train_kg_dict, train_kg = self.construct_data(n_tim_rel, tim_dis_dict)

        # store kg data
        # self.kg['utp'] = self.filtering_triple(utp_triple)  
        # self.kg['ptp'] = self.filtering_triple(ptp_triple)
        self.kg['ptp_dict'] =tim_dis_dict
        self.kg['poi_trans'] = self.temp_dis_dict
        self.kg['timining_rel'] =list(set(self.tim))
        self.kg['tim_rel'] = list(set(self.tim_rel))
        self.kg['dis_rel'] = list(set(self.dis_rel))
        self.kg['train_kg'] = np.concatenate([train_kg['utp'],train_kg['ptp']])
        self.kg['train_kg_dict'] = train_kg_dict
        # self.kg['train_utp'] = self.train_utp
        # self.kg['train_ptp'] = self.train_ptp
        self.kg['max_dis_tim']=[self.tim_max,self.dis_max]
        


    # ############# 6. save variables
    def get_parameters(self):
        parameters = {}
        parameters['TWITTER_PATH'] = self.TWITTER_PATH
        parameters['SAVE_PATH'] = self.SAVE_PATH
        parameters['trace_len_min'] = self.trace_len_min
        parameters['location_global_visit_min'] = self.location_global_visit_min
        parameters['hour_gap'] = self.hour_gap
        parameters['min_gap'] = self.min_gap
        parameters['session_max'] = self.session_max
        parameters['filter_short_session'] = self.filter_short_session
        parameters['sessions_min'] = self.sessions_count_min
        parameters['train_split'] = self.train_split

        return parameters

    def save_variables(self):
        foursquare_dataset = {'data_neural': self.data_neural, 'vid_list': self.vid_list, 'uid_list': self.uid_list,
                              'parameters': self.get_parameters(), 'data_filter': self.data_filter,
                              'vid_lookup': self.vid_lookup, 'KG':self.kg}
        pickle.dump(foursquare_dataset, open(self.SAVE_PATH + self.save_name + '.pkl', 'wb'))
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace_min', type=int, default=10, help="raw trace length filter threshold")
    parser.add_argument('--global_visit', type=int, default=10, help="location global visit threshold")
    parser.add_argument('--hour_gap', type=int, default=24, help="maximum interval of two trajectory points")
    parser.add_argument('--min_gap', type=int, default=10, help="minimum interval of two trajectory points")
    parser.add_argument('--session_max', type=int, default=10, help="control the length of session not too long")
    parser.add_argument('--session_min', type=int, default=3, help="control the length of session not too short")
    parser.add_argument('--sessions_min', type=int, default=5, help="the minimum amount of the good user's sessions")
    parser.add_argument('--train_split', type=float, default=0.8, help="train/test ratio")
    parser.add_argument('--time_gap', type=float, default=3600, help="time gap")
    parser.add_argument('--distance_gap', type=float, default=0.06, help="distance_gap")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data_generator = DataFoursquare(trace_min=args.trace_min, global_visit=args.global_visit,
                                    hour_gap=args.hour_gap, min_gap=args.min_gap,
                                    session_min=args.session_min, session_max=args.session_max,
                                    sessions_min=args.sessions_min, train_split=args.train_split,
                                    time_split=args.time_gap, distance_split=args.distance_gap)
    parameters = data_generator.get_parameters()
    print('############PARAMETER SETTINGS:\n' + '\n'.join([p + ':' + str(parameters[p]) for p in parameters]))
    print('############START PROCESSING:')
    print('load trajectory from {}'.format(data_generator.TWITTER_PATH))
    data_generator.load_trajectory_from_tweets()
    print('filter users')
    data_generator.filter_users_by_length()
    print('build users/locations dictionary')
    data_generator.build_users_locations_dict()
    # data_generator.load_venues()
    data_generator.venues_lookup()  
    print('prepare data for neural network')
    data_generator.prepare_neural_data()
    data_generator.prepare_kg_data()
    print('save prepared data')
    data_generator.save_variables()
    print('raw users:{} raw locations:{}'.format(
        len(data_generator.data), len(data_generator.venues)))
    print('final users:{} final locations:{} '.format(
        len(data_generator.data_neural), len(data_generator.vid_list)))
    print('check_in num:{} sessions length:{} '.format(
        data_generator.check_in_num, data_generator.sessions_num))
    print('triple num:{} '.format(
        len(data_generator.kg['utp'])+len(data_generator.kg['ptp'])))
    
