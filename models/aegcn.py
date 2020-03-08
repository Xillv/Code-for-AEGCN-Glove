# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
from layers.squeeze_embedding import SqueezeEmbedding
from layers.attention import Attention
from layers.point_wise_feed_forward import PositionwiseFeedForward_GCN

class GraphConvolution(nn.Module):
    """
    Similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, opt, bias=True):
        super(GraphConvolution, self).__init__()
        self.opt = opt
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))

        self.ffn_gcn = PositionwiseFeedForward_GCN(opt.hidden_dim*2, dropout=opt.dropout)    #
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        # output = self.attn_g(output,output)
        output = self.ffn_gcn(output)   #

        if self.bias is not None:
            return output + self.bias
        else:
            return output

class AEGCN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(AEGCN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding()  #
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.aspect_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.attn_k = Attention(opt.embed_dim * 2, out_dim=opt.hidden_dim, n_head=opt.head, score_function='mlp',
                                dropout=opt.dropout)  #
        self.attn_a = Attention(opt.embed_dim * 2, out_dim=opt.hidden_dim, n_head=opt.head, score_function='mlp',
                                dropout=opt.dropout)  #
        # self.attn_s1 = Attention(opt.embed_dim*2, out_dim=opt.hidden_dim, n_head=3, score_function='mlp', dropout=0.5)

        self.attn_q = Attention(opt.embed_dim*2, out_dim=opt.hidden_dim, n_head=opt.head, score_function='mlp',
                                dropout=opt.dropout)  #

        self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim, opt)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim, opt)
        self.attn_k_q = Attention(opt.hidden_dim, n_head=opt.head, score_function='mlp', dropout=opt.dropout) #
        self.attn_k_a = Attention(opt.hidden_dim, n_head=opt.head, score_function='mlp', dropout=opt.dropout)
        #self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)

        self.text_embed_dropout = nn.Dropout(opt.dropout)
        self.aspect_embed_dropout = nn.Dropout(opt.dropout)

        self.dense = nn.Linear(opt.hidden_dim * 3, opt.polarities_dim)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i,0]):
                weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i,1]+1, text_len[i]):
                weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.opt.device)
        return weight*x


    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj = inputs
        #context_len = torch.sum(text_indices != 0, dim=-1)   #
        #target_len = torch.sum(aspect_indices != 0, dim=-1)  #
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        # text = self.squeeze_embedding(text, text_len)
        text = self.text_embed_dropout(text)
        aspect = self.embed(aspect_indices)    #
        # aspect = self.aspect_embed_dropout(aspect)    #
        aspect = self.squeeze_embedding(aspect, aspect_len)
        text_out, (_, _) = self.text_lstm(text, text_len)
        aspect_out, (_, _) = self.aspect_lstm(aspect, aspect_len)  #add aspect
        hid_context = self.squeeze_embedding(text_out, text_len)   #
        hid_aspect = self.squeeze_embedding(aspect_out, aspect_len)  #




        hc, _ = self.attn_k(hid_context, hid_context)   #


        ha, _ = self.attn_a(hid_aspect, hid_aspect)




        x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))

        x = self.squeeze_embedding(x, text_len)
        hg, _ = self.attn_q(x, x)


        hc_hg, _ = self.attn_k_q(hc, hg)

        hg_ha, _ = self.attn_k_a(hg, ha)

        text_len = torch.tensor(text_len, dtype=torch.float).to(self.opt.device)
        aspect_len = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)

        hc_mean = torch.div(torch.sum(hc, dim=1), text_len.view(text_len.size(0), 1))

        hc_hg_mean = torch.div(torch.sum(hc_hg, dim=1), text_len.view(text_len.size(0), 1))

        hg_ha_mean = torch.div(torch.sum(hg_ha, dim=1), aspect_len.view(text_len.size(0), 1))

        final_x = torch.cat((hc_hg_mean,hc_mean, hg_ha_mean), dim=-1)


        output = self.dense(final_x)
        return output

