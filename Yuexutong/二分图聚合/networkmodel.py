import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.utils import expand_as_pair

import dgl.function as fn

import numpy as np
import pandas as pd

class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        #init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])
        
        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float()) #一维
        self.phase = nn.Parameter(torch.zeros(time_dim).float())  #size = (1,time_dim)
        
        #self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)

        #torch.nn.init.xavier_normal_(self.dense.weight)
        
    def forward(self, ts_1):
        ts = ts_1
        # ts: [N, L]
        # batch_size = ts.size(0)
        seq_len = ts.size(0)
                
        ts = ts.view(seq_len, 1)# [L, 1]  basic_freq [time_dim]  一维  basis_freq.view(1, 1, -1):[1,1,time_dim]
        map_ts = ts * self.basis_freq.view(1, -1) # [L, time_dim]
        map_ts += self.phase.view(1, -1)
        
        harmonic = torch.cos(map_ts)

        return harmonic #self.dense(harmonic)

class GNet(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers):  #out_size = 2
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if num_layers == 1:
                self.layers.append(SAGEConv(in_size, hid_size, out_size))
            else:
                if i == 0:
                    self.layers.append(SAGEConv(in_size, hid_size, hid_size))
                elif i == num_layers - 1:
                    self.layers.append(SAGEConv(hid_size, hid_size, out_size))
                else:
                    self.layers.append(SAGEConv(hid_size, hid_size, hid_size))
        self.dropout = nn.Dropout(0.1)

    def forward(self, blocks,h):   #在main中引用的时候使用model（blocks，featrue）  blocks是小批量子图块  featreus是源节点特侦
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h.data)
        return h


class SAGEConv(nn.Module):
    def __init__(self, in_size, hid_size, out_size, activation=F.relu, dropout=0.2,attn_drop=0.2):
        super().__init__()

        self.activation = activation
        self.text_proj = dglnn.linear.TypedLinear(in_size, hid_size, 4, regularizer='basis', num_bases=3)  #将文本特征 用户特征维度对齐
        #dglnn.linear.TypedLinear（input_feat,out_size,num_type,regularizer,num_base) 再引用的时候传入proj(x,x_type),x是样本二维张量，x_type是一个一维张量，并且与x的行数一样，一一对应x样本对应的类别
        self.user_proj = dglnn.linear.TypedLinear(in_size, in_size, 4, regularizer='basis', num_bases=3)  #目标节点本身
        self.linear = nn.Linear(hid_size*2, out_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(768)
        self.hid_size = hid_size
        self.attn_drop = nn.Dropout(attn_drop)
        self.time_encoding = TimeEncode(hid_size)
        self.dst_weight = nn.Linear(768,768,bias=False)
        self.src_weight = nn.Linear(768,768,bias=False)
        attn_drop = nn.Dropout(0.2)
        self.dst_proj = dglnn.linear.TypedLinear(768, 768, 4, regularizer='basis', num_bases=3)
        self.src_proj = dglnn.linear.TypedLinear(768, 768, 4, regularizer='basis', num_bases=3)
        # self.dropout = nn.Dropout(0.2)
        self.agg_gate_i = nn.Linear(hid_size * 2, 768, bias=False)
    
    def message_func(self,edges):
        dic = {}
        
        dic['time'] = edges.data['etime']
        dic['first_etime'] = edges.data['first_etime'] #一维
        dic['second_etime'] = edges.data['second_etime']
        dic['first_etype'] = edges.data['first_etype'] #一维 表示所有目标节点所连接的边
        dic['second_etype'] = edges.data['second_etype']
        dic['etype'] = edges.data['etype']
        dic['src_f'] = edges.src['n'] #二维
        dic['dst_f'] = edges.dst['n']
        dic['hete'] = edges.data['hete']
        dic['homo'] = edges.data['homo']
        return dic
    
    def reduce_func(self,nodes): 
        idx = 0
       

        #一跳邻居信息汇总
        idx = (nodes.mailbox['hete'] == 1)
        h_hetes = []
        for i in range(len(nodes.mailbox['hete'])): #只针对nodes。mailbox中一个节点
            
            hete_src_info = nodes.mailbox['src_f'][i][idx[i]] #二维 表示节点一的信息
            time_hete_info = nodes.mailbox['time'][i][idx[i]]  #一维 表示节点一的所有邻居节点时间
            hete_dst_info = nodes.mailbox['dst_f'][i][idx[i]] #二维 所有接受信息的节点一
            # print("节点1",hete_src_info,hete_src_info.shape)
            hete_src_info = self.activation(self.src_proj(self.dropout(self.layer_norm(hete_src_info)),nodes.mailbox['first_etype'][i][idx[i]]))
        
 
            
            #一阶邻居信息聚合
            eij_hete = torch.sum((self.time_encoding(time_hete_info) + hete_src_info) * hete_dst_info, dim=1)\
                        /torch.sqrt(torch.tensor(768).float())  #一维
           
            alpha = self.attn_drop(F.softmax(eij_hete,dim=0))#一维
           
            h_hete = torch.sum(alpha.unsqueeze(-1)*(self.time_encoding(time_hete_info) + hete_src_info),dim=0)
            #h聚合每一个节点的一跳邻居信息
            
            
            h_hetes.append(h_hete)
        
        h_hete_info = torch.stack(h_hetes) #二维
        
        
        #二跳邻居信息汇总

        idx_1 = (nodes.mailbox['hete'] == 0)
        h_homos = []

        for j in range(len(nodes.mailbox['hete'])):
            
            homo_src_info = nodes.mailbox['src_f'][j][idx_1[j]]
            time_first_info = nodes.mailbox['first_etime'][j][idx_1[j]]
            time_second_info = nodes.mailbox['second_etime'][j][idx_1[j]]
            etype_first_info = nodes.mailbox['first_etype'][j][idx_1[j]]
            etype_second_info = nodes.mailbox['second_etype'][j][idx_1[j]]
            homo_dst_info = nodes.mailbox['dst_f'][j][idx_1[j]]
        
        #二跳邻居信息聚合
            dst_feat = self.activation(self.dst_proj(self.dropout(self.layer_norm(homo_dst_info)),etype_first_info))
        # print('dst_feat',dst_feat)
            homo_feat = self.activation(self.user_proj(self.dropout(self.layer_norm(homo_src_info)),etype_second_info))
        # print('src',homo_feat)
            time_homo = torch.max(time_first_info,time_second_info)
            # print('time',time_homo)
            eij_homo = torch.sum((self.time_encoding(time_homo) + homo_feat) * dst_feat, dim=1)\
                        /torch.sqrt(torch.tensor(768).float())  #二维
            alpha = self.attn_drop(F.softmax(eij_homo,dim=0)) #一维
            h_homo = torch.sum(alpha.unsqueeze(-1)*(self.time_encoding(time_homo) + homo_feat),dim=0)
            # print('h_homo',h_homo,h_homo.shape)
            h_homos.append(h_homo)
        h_homo_info = torch.stack(h_homos)
        # print(f"cat{j}",torch.cat([h_hete_info,h_homo_info],dim=-1).shape)

        return {'n': torch.cat([h_hete_info,h_homo_info],dim=-1)}

    

    def forward(self, g, h):  #forward输入graph，feat
        h_src, h_dst = expand_as_pair(h, g)   #源节点和目标节点  指定图类型，然后根据图类型扩展输入特征

        with g.local_scope():   #输入图对象的规范性检测
            
            
            g.srcdata['n'] = self.src_weight(h_src)
            g.dstdata['n'] = self.dst_weight(h_dst)
            
            g.edata['count'] = g.edata['count'].float()  #g.edata返回边的特征
            g.update_all(self.message_func, self.reduce_func)
            # g.update_all(fn.copy_e('count', 'm'), fn.sum('m', 'w'))
            

            n = g.dstdata['n']
            
            z = self.activation(self.linear(n))

            z_norm = z.norm(2, 1, keepdim=True) 
            z_norm = torch.where(z_norm == 0, torch.tensor(1.).to(z_norm), z_norm)
            z = z / z_norm


            return z
