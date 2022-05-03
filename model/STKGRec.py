import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)

class STKGRec(nn.Module):
    def __init__(self, args, n_users, n_locs,n_tim_rel, n_tim_dis_gap, dropout=0.8, 
    user_dropout=0.5, data_neural = None, kg = None):
        super(self.__class__, self).__init__()
        self.n_users = n_users
        self.n_locs = n_locs
        self.n_tim_rel = n_tim_rel
        self.n_tim_dis_gap = n_tim_dis_gap
        self.hidden_units = args['hidden_size']
        if args['hidden_size'] == None:
            self.emb_size = args['hidden_size']
        self.emb_size = args['hidden_size']
        self.entities = nn.Embedding(self.n_users+self.n_locs, self.emb_size)
        self.relations = nn.Embedding(self.n_tim_rel+self.n_tim_dis_gap, self.emb_size)
        self.gru = nn.GRU(input_size = self.emb_size, hidden_size = self.hidden_units)
        self.gru_history = nn.GRU(input_size= self.emb_size, hidden_size = self.hidden_units)
        self.rel_rnn = nn.GRU(input_size=self.emb_size, hidden_size = self.hidden_units)# could be the same as self.gru
        self.dilated_rnn = nn.GRUCell(input_size=self.emb_size, hidden_size = self.hidden_units)
        self.dropout = nn.Dropout(0.0)
        self.user_dropout = nn.Dropout(user_dropout)
        self.data_neural = data_neural
        self.kg = kg
        self.tim_dis_dict = self.kg['ptp_dict']
        self.linear = nn.Linear(self.hidden_units*4 , self.n_locs)
        self.linear1 = nn.Linear(3*self.emb_size, self.emb_size)
        self.rel_W = nn.Linear(2*self.hidden_units,self.hidden_units)
        self.trans_rel = nn.Linear(2*self.hidden_units,self.hidden_units)
        self.trans_att = nn.Linear(self.hidden_units,self.hidden_units)
        self.init_weights() 
        self.W_R = nn.Parameter(torch.Tensor(self.n_tim_rel+self.n_tim_dis_gap,  self.emb_size,  self.emb_size))
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))
        self.kg_l2loss_lambda = args['kg_lambda']

    def init_weights(self):
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)

    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relations(r)                 # (kg_batch_size, relation_dim)
        W_r = self.W_R[r]                                # (kg_batch_size, entity_dim, relation_dim)

        h_embed = self.entities(h)              # (kg_batch_size, entity_dim)
        pos_t_embed = self.entities(pos_t)      # (kg_batch_size, entity_dim)
        neg_t_embed = self.entities(neg_t)      # (kg_batch_size, entity_dim)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)             # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)     # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)     # (kg_batch_size, relation_dim)

        # Equation (1)
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)     # (kg_batch_size)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)     # (kg_batch_size)

        # Equation (2)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss



    def sessions_score(self, user_vectors, item_vectors, mask_batch_ix_non_local, session_id_batch, 
                tim_batch,tim_dis_gap_batch,device, is_train,sequence_dilated_rnn_index_batch):
        batch_size = item_vectors.size()[0]
        sequence_size = item_vectors.size()[1]
        locs = self.entities(item_vectors)  #[batch_size, sequence_length, embedding_dim]
        item_vectors = item_vectors.cpu()    #[batch_size,sequence_length,]
        users = user_vectors.cpu()
        users = users.data.numpy().tolist()
        users_seq = [[users[id]]*sequence_size for id in range(batch_size)]
        users_seq = Variable(torch.LongTensor(np.array(users_seq))).to(device)
        vec_users = self.entities(users_seq)
        users_vec = self.entities(user_vectors)
        tims = self.relations(tim_batch)    
        # tim_dis_gap = self.relations(tim_dis_gap_batch)
        tim_loc = locs + tims + vec_users  #[batch_size, sequence_length, embedding_dim]
        x = tim_loc
        x = x.transpose(0,1)
        h1 = Variable(torch.zeros(1, batch_size, self.hidden_units)).cuda()  
        out, h1 =self.gru(x, h1)
        out = out.transpose(0,1)
        x1 = tim_loc

        # ###########################################################
        user_batch = np.array(user_vectors.cpu())
        y_list = []
        out_hie = []
        out_trans = []
        for ii in range(batch_size):
            current_session_input_dilated_rnn_index = sequence_dilated_rnn_index_batch[ii]
            hiddens_current = x1[ii]  #[sequence_length * embedding_dim]
            dilated_gru_outs_h = []  
            # dilated_lstm_outs_c = []
            for index_dilated in range(len(current_session_input_dilated_rnn_index)):  
                index_dilated_explicit = current_session_input_dilated_rnn_index[index_dilated]
                hidden_current = hiddens_current[index_dilated].unsqueeze(0) 
                if index_dilated == 0:
                    h = Variable(torch.zeros(1, self.hidden_units)).cuda()  
                    h = self.dilated_rnn(hidden_current, h)
                    dilated_gru_outs_h.append(h) 
                else:
                    h = self.dilated_rnn(hidden_current, dilated_gru_outs_h[index_dilated_explicit])
                    dilated_gru_outs_h.append(h)
            dilated_gru_outs_h.append(hiddens_current[len(current_session_input_dilated_rnn_index):]) 
            dilated_out = torch.cat(dilated_gru_outs_h, dim = 0).unsqueeze(0)   
            out_hie.append(dilated_out)
        # ###########################################################
            out_rel = []
            user_id_current = user_batch[ii]-self.n_locs  
            tim_dis_gap_emb = self.relations(tim_dis_gap_batch[ii][:-1])
            user_emb = users_vec[ii].unsqueeze(0)
            current_tim_dis_rel= torch.cat((user_emb,tim_dis_gap_emb))
            session_id_current = session_id_batch[ii] 
            current_session_embed = out[ii] 
            current_session_mask = mask_batch_ix_non_local[ii].unsqueeze(1) 
            sequence_length = int(sum(np.array(current_session_mask.cpu()))[0])
            current_session_represent_list = []
            current_relation_list =[]
            if is_train:
                for iii in range(sequence_length-1): 
                    current_session_represent = torch.sum(current_session_embed * current_session_mask, dim=0).unsqueeze(0)/sum(current_session_mask) #[1 * embedding_dim]
                    current_session_represent_list.append(current_session_represent)
                    current_relation = current_tim_dis_rel[0:iii+1]
                    current_relation = torch.sum(current_relation, dim=0).unsqueeze(0)
                    current_relation_list.append(current_relation)
            else:
                for iii in range(sequence_length-1):
                    current_session_represent_rep_item = current_session_embed[0:iii+1]
                    current_session_represent_rep_item = torch.sum(current_session_represent_rep_item, dim = 0).unsqueeze(0)/(iii + 1)
                    current_session_represent_list.append(current_session_represent_rep_item)
                    current_relation = current_tim_dis_rel[0:iii+1]
                    current_relation = torch.sum(current_relation, dim=0).unsqueeze(0)
                    current_relation_list.append(current_relation)
            current_session_represent = torch.cat(current_session_represent_list, dim = 0) # [sequence_length-1 * embedding_dim]
            current_rel_present = torch.cat(current_relation_list,dim=0)
            current = current_session_embed[:sequence_length - 1]
            current_rel_att =F.softmax(torch.mm(current,current_rel_present.transpose(0,1)))
            current = torch.mm(current_rel_att,current)
            out_rel_current_padd = Variable(torch.FloatTensor(sequence_size - sequence_length + 1, self.emb_size).zero_(),requires_grad=False).cuda()
            out_rel.append(current)
            out_rel.append(out_rel_current_padd)
            out_rel_list = torch.cat(out_rel, dim=0).unsqueeze(0)
            out_trans.append(out_rel_list)
            

            session_list = []
            user_id_vec = users_vec[ii]
            for jj in range(session_id_current):
                rel_gru_h = []
                trans_rel = []
                sequence_loc = [s[0] for s in self.data_neural[user_id_current]['sessions'][jj]]
                sequence_len = len(sequence_loc)
                sequence_tim = [s[1] for s in self.data_neural[user_id_current]['sessions'][jj]]
                sequence_tim_dis_gap = [int(self.tim_dis_dict[tuple(s[1])]+self.n_tim_rel) for s in self.data_neural[user_id_current]['sessions_trans'][jj]]
                sequence_loc = Variable(torch.LongTensor(np.array(sequence_loc))).cuda()
                sequence_tim = Variable(torch.LongTensor(np.array(sequence_tim))).cuda()
                sequence_tim_dis_gap = Variable(torch.LongTensor(np.array(sequence_tim_dis_gap))).cuda()

                sequence_loc = self.entities(sequence_loc)
                sequence_tim = self.relations(sequence_tim)
                sequence_tim_dis_gap = self.relations(sequence_tim_dis_gap)
                rel_padd = Variable(torch.FloatTensor(1, self.emb_size).zero_(),requires_grad=False).cuda()
                sequence_tim_dis = torch.cat([sequence_tim_dis_gap[:sequence_len-1],rel_padd],dim=0)
                trans_vec = (sequence_loc+sequence_tim_dis)
                sequence_user = user_id_vec.unsqueeze(0).expand(sequence_len, self.emb_size)
                user_loc_tim_vec = (sequence_user + sequence_loc+sequence_tim+sequence_tim_dis)
                att_seq = F.softmax(torch.mm(trans_vec, user_loc_tim_vec.transpose(0,1)))
                seq_rel = torch.mm(att_seq,user_loc_tim_vec).unsqueeze(1)
                h2 = Variable(torch.zeros(1,1, self.hidden_units)).cuda()  #[1, embedding_dim]
                out2,h2 = self.rel_rnn(seq_rel, h2)
                seq_W =  Variable(torch.FloatTensor(sequence_length-1,sequence_len)).cuda()
                nn.init.xavier_uniform(seq_W)
                hidden_sequence = torch.mm(seq_W, out2.squeeze())
                session_list.append(hidden_sequence.unsqueeze(0))
            sessions_represent = torch.cat(session_list, dim=0).transpose(0,1)  #[sequence_length, len(session), embedding_dim]
            current_session_represent = current_session_represent.unsqueeze(2)  #[sequence_length, embedding_dim , 1]
            sims = F.softmax(sessions_represent.bmm(current_session_represent).squeeze(2), dim = 1).unsqueeze(1) ##==> sequence_length * 1 * 1
            out_y_current = sims.bmm(sessions_represent).squeeze(1)#[sequence_length* embedding_dim]
            ######################################################################################
            out_y_current_padd = Variable(torch.FloatTensor(sequence_size - sequence_length + 1, self.emb_size).zero_(),requires_grad=False).cuda()
            out_layer_2_list = []
            out_layer_2_list.append(out_y_current)
            out_layer_2_list.append(out_y_current_padd)  
            out_layer_2 = torch.cat(out_layer_2_list,dim = 0).unsqueeze(0) 
            y_list.append(out_layer_2)
        y = torch.cat(y_list, dim=0)
        out = F.selu(out)
        out_hie = F.selu(torch.cat(out_hie, dim = 0))  
        out_trans = F.selu(torch.cat(out_trans, dim = 0))
        out_put_emb_v1 = torch.cat([y, out,out_hie,out_trans], dim=2) 
        output_ln = self.linear(out_put_emb_v1)  
        output = F.log_softmax(output_ln, dim=-1)
        return output

    
    def forward(self, mode, *input):
        if mode == 'calc_kg_loss':
            return self.calc_kg_loss(*input)
        if mode == 'predict':
            return self.sessions_score(*input)