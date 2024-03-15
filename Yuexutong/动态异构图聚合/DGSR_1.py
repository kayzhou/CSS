
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math


def get_sent_features(model_name, sents, device):
    model = SentenceTransformer(model_name)
    return model.encode(sents, batch_size=32, device=device, show_progress_bar=True)

def get_user_features(model_name, users, device):
    model = SentenceTransformer(model_name)
    return model.encode(users, batch_size=32, device=device, show_progress_bar=True)

class DGSR(nn.Module):
    def __init__(self, user_num, item_num, input_dim, item_max_length, user_max_length, feat_drop=0.2, attn_drop=0.2,
                 last_item=True, layer_num=3, time=True):
        super(DGSR, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.hidden_size = input_dim
        self.item_max_length = item_max_length
        self.user_max_length = user_max_length
        self.layer_num = layer_num
        self.time = time
        self.last_item = last_item

        self.user_embedding = nn.Embedding(self.user_num, self.hidden_size) #用户节点初始化嵌入   nn.Embedding(num_embedding,embedding_dim)  （嵌入字典大小（多少词），每个词词向量维度）
        self.item_embedding = nn.Embedding(self.item_num, self.hidden_size)
        if self.last_item:  #最后一层融合了所有层的向量
            self.unified_map = nn.Linear((self.layer_num + 1) * self.hidden_size, self.hidden_size, bias=False)
        else:
            self.unified_map = nn.Linear(self.layer_num * self.hidden_size, self.hidden_size, bias=False)
        self.layers = nn.ModuleList([DGSRLayers(self.hidden_size, self.hidden_size, self.user_max_length, self.item_max_length, feat_drop, attn_drop) for _ in range(self.layer_num)])
        self.reset_parameters()
        
        self.dense = nn.Linear(self.hidden_size,3)
        self.activation = F.relu

    def forward(self, g, user_index=None, last_item_index=None, neg_tar=None, is_training=False):
        feat_dict = None
        user_layer = []
        g.nodes['user'].data['user_h'] = self.user_embedding(g.nodes['user'].data['user_id'].cuda())   #特征初始化
        g.nodes['item'].data['item_h'] = self.item_embedding(g.nodes['item'].data['item_id'].cuda())
        if self.layer_num > 0:
            for conv in self.layers: 
                feat_dict = conv(g, feat_dict)   #DGSPlayer得到的结果存储为一个字典
                user_layer.append(graph_user(g, user_index, feat_dict['user']))
            if self.last_item:
                item_embed = graph_item(g, last_item_index, feat_dict['item'])
                user_layer.append(item_embed)
        unified_embedding = self.unified_map(torch.cat(user_layer, -1))  #拼接所有层的输出
        
        output = self.activation(self.dense(unified_embedding))

        return output #二维[n,3]

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for weight in self.parameters():
            if len(weight.shape) > 1:
                nn.init.xavier_normal_(weight, gain=gain)



class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        #init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])
        
        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = nn.Parameter(torch.zeros(time_dim).float())  #size = (1,time_dim)

        
    def forward(self, ts_1):
        ts = ts_1
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)
                
        ts = ts.view(batch_size, seq_len, 1)# [N, L, 1]  basic_freq [time_dim]  一维  basis_freq.view(1, 1, -1):[1,1,time_dim]
        map_ts = ts * self.basis_freq.view(1, 1, -1) # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)
        
        harmonic = torch.cos(map_ts)

        return harmonic.squeeze(0) #self.dense(harmonic)

class DGSRLayers(nn.Module):
    def __init__(self, in_feats, out_feats, user_max_length, item_max_length, feat_drop=0.2, attn_drop=0.2, K=4):
        super(DGSRLayers, self).__init__()
        self.hidden_size = in_feats
        self.user_max_length = user_max_length
        self.item_max_length = item_max_length
        self.K = torch.tensor(K).cuda()
        self.agg_gate_u = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.agg_gate_i = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.norm_user = nn.LayerNorm(self.hidden_size)
        self.norm_item = nn.LayerNorm(self.hidden_size)
        self.feat_drop = nn.Dropout(feat_drop)
        self.atten_drop = nn.Dropout(attn_drop)
        self.user_weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.item_weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.user_update = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        self.item_update = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)

        
        # attention+ attention mechanism
        if self.user_short in ['last', 'att']:
            self.last_weight_u = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        if self.item_short in ['last', 'att']:
            self.last_weight_i = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        
        #时间编码
        if self.item_long in ['orgat']:
            self.i_time_encoding = TimeEncode(self.hidden_size)  #（user_num,dim)
            self.i_time_encoding_k = TimeEncode(self.hidden_size)
        if self.user_long in ['orgat']:
            self.u_time_encoding = TimeEncode(self.hidden_size)
            self.u_time_encoding_k = TimeEncode(self.hidden_size)


    def user_update_function(self, user_now, user_old):  #用户节点的更新规则
        return F.elu(self.user_update(torch.cat([user_now, user_old], -1)))

    def item_update_function(self, item_now, item_old):
        return F.elu(self.item_update(torch.cat([item_now, item_old], -1)))

    def forward(self, g, feat_dict=None):

        user_ = feat_dict['user'].cuda()  #使用指定的用户节点初始化特征  user_old
        item_ = feat_dict['item'].cuda()
 
        g.nodes['user'].data['user_h'] = self.user_weight(self.feat_drop(user_)) #线性层   user_now
        g.nodes['item'].data['item_h'] = self.item_weight(self.feat_drop(item_))
        g = self.graph_update(g)
        g.nodes['user'].data['user_h'] = self.user_update_function(g.nodes['user'].data['user_h'], user_)
        g.nodes['item'].data['item_h'] = self.item_update_function(g.nodes['item'].data['item_h'], item_)
        f_dict = {'user': g.nodes['user'].data['user_h'], 'item': g.nodes['item'].data['item_h']}
        return f_dict

    def graph_update(self, g):

        g.multi_update_all({'by': (self.user_message_func, self.user_reduce_func),  #‘by’整合用户关系  item-by-user
                            'pby': (self.item_message_func, self.item_reduce_func)}, 'sum') #sum表示不同关系的聚合函数
        return g
        #g.multi_update_all异构图信息传递函数  先对每种关系做信息传递，再跨关系传递信息
        #接受一个字典  字典的健是不同的关系，值得对应的这种关系的整合函数（updaall;更新所有终点类型的节点而不是所有的节点） 

    def item_message_func(self, edges):  #消息函数  生成消息m 存放在mailbox中
        dic = {}
        dic['time'] = edges.data['time']
        dic['user_h'] = edges.src['user_h']   # user-pby-item   edge.src是user dst是item
        dic['item_h'] = edges.dst['item_h']
        return dic      #使用边的属性生成消息 生成的也是边的一些信息字段  时间啊 用户item-h这些东西  存放在目标节点的mailbox中             

    def item_reduce_func(self, nodes):  #聚合函数   user-pby-item  目标节点是item
        h = []
        #先根据time排序
        #order = torch.sort(nodes.mailbox['time'], 1)[1]
        time = nodes.mailbox['time']

        length = nodes.mailbox['item_h'].shape[0]

        e_ij = torch.sum((self.i_time_encoding(time) + nodes.mailbox['user_h']) * nodes.mailbox['item_h'], dim=2)\
                /torch.sqrt(torch.tensor(self.hidden_size).float()) #时间差

        alpha = self.atten_drop(F.softmax(e_ij, dim=1))
        if len(alpha.shape) == 2:
            alpha = alpha.unsqueeze(2)
        h_long = torch.sum(alpha * (nodes.mailbox['user_h'] + self.i_time_encoding_k(time)), dim=1) #时间差
        h.append(h_long)

    def user_message_func(self, edges):
        dic = {}
        dic['time'] = edges.data['time']
        dic['item_h'] = edges.src['item_h']  #源节点
        dic['user_h'] = edges.dst['user_h']  #目标节点
        return dic

    def user_reduce_func(self, nodes):
        h = []
        time = nodes.mailbox['time']

        length = nodes.mailbox['user_h'].shape[0]
       
        e_ij = torch.sum((self.u_time_encoding(time) + nodes.mailbox['item_h']) *nodes.mailbox['user_h'],
                        dim=2) / torch.sqrt(torch.tensor(self.hidden_size).float())  #时间差编码
   
        alpha = self.atten_drop(F.softmax(e_ij, dim=1))
        if len(alpha.shape) == 2:
            alpha = alpha.unsqueeze(2)
        h_long = torch.sum(alpha * (nodes.mailbox['item_h'] + self.u_time_encoding_k(time)), dim=1) #时间差编码
        h.append(h_long)
        
        if len(h) == 1:
            return {'user_h': h[0]}
        else:
            return {'user_h': self.agg_gate_u(torch.cat(h,-1))}

def graph_user(bg, user_index, user_embedding):  #在上面调用的时候graph_user(g, user_index, feat_dict['user']
    b_user_size = bg.batch_num_nodes('user')  #user节点的个数

    tmp = torch.roll(torch.cumsum(b_user_size, 0), 1)  #tensor.roll()是将张量沿着第以为向下滚动亦歌，tensor.cumsum()沿着指定dim元素累加  返回的是一个具有相同shape的张量
    tmp[0] = 0
    new_user_index = tmp + user_index
    return user_embedding[new_user_index]

def graph_item(bg, last_index, item_embedding):
    b_item_size = bg.batch_num_nodes('item')

    tmp = torch.roll(torch.cumsum(b_item_size, 0), 1)
    tmp[0] = 0
    new_item_index = tmp + last_index
    return item_embedding[new_item_index]

def order_update(edges):
    dic = {}
    dic['order'] = torch.sort(edges.data['time'])[1]
    dic['re_order'] = len(edges.data['time']) - dic['order']
    return dic


def collate(data):
    user_l = []
    graph = []
    label = []
    for da in data:
        user_l.append(da[1]['u_alis'])
        graph.append(da[0][0])
        label.append(da[1]['target'])
    return torch.tensor(user_l).long(), dgl.batch(graph), torch.tensor(label).long()


def neg_generate(user, data_neg, neg_num=10):  ##user_neg(data, item_num)  data:dataframe  data——neg是一个series  索引是每个用户id，值是所有负样本列表
    neg = np.zeros((len(user), neg_num), np.int32)
    for i, u in enumerate(user):
        neg[i] = np.random.choice(data_neg[u], neg_num, replace=False)
    return neg
 

def collate_test(data, user_neg):
    # 生成负样本和每个序列的长度
    user_1 = []
    graph = []
    label = []
    for da in data:
        user_1.append(da[1]['u_alis'])
        graph.append(da[0][0])
        label.append(da[1]['target'])
    return torch.tensor(user_1).long(), dgl.batch(graph), torch.tensor(label).long(),torch.Tensor(neg_generate([t.item() for t in user_1], user_neg)).long()
