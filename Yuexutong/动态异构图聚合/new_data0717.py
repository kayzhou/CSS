import dgl
import pandas as pd
import numpy as np
import datetime
import argparse
from dgl.sampling import sample_neighbors, select_topk
import torch
import os
from dgl import save_graphs
from joblib import Parallel, delayed
from sentence_transformers import SentenceTransformer
import pickle
import random


# 计算item序列的相对次序
def cal_order(data):
    data = data.sort_values(['time'], kind='mergesort') #排序
    data['order'] = range(len(data))
    return data

# 以itm为锚节点计算user序列的相对次序
def cal_u_order(data):
    data = data.sort_values(['time'], kind='mergesort')
    data['u_order'] = range(len(data))
    return data
import numpy as np
import dgl
import dgl.backend as F
from dgl.dataloading import BlockSampler
from dgl.sampling.randomwalks import random_walk
import torch


import dgl.backend as F1
class NeighborSampler(BlockSampler):
    def __init__(self, random_walk_length, num_random_walks, num_neighbors, num_layers, termination_prob=0.5, num_traversals=1):
        super().__init__()
        
        self.random_walk_length = random_walk_length
        self.num_random_walks = num_random_walks
        self.num_neighbors = num_neighbors
        self.num_traversals = num_traversals
        self.num_layers = num_layers

        restart_prob = np.zeros(random_walk_length * num_traversals)
        restart_prob[random_walk_length::random_walk_length] = termination_prob
        self.restart_prob = F1.tensor(restart_prob, dtype=F1.float32)

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):   #sample_blocks(）从最后一层开始生成一个块的列表
        output_nodes = seed_nodes
        blocks = []
        for _ in range(self.num_layers):
            frontier = self.sample_neighbors(g, seed_nodes)  #frontier是邻居子图
            block = dgl.to_block(frontier, seed_nodes) #将邻居子图转变成块 seed_nodes就是块的目标节点
            seed_nodes = block.srcdata[dgl.NID] #在将块的input节点作为目标节点进行下一次采样
            blocks.insert(0, block) #在block列表中0位置在添加一个block对象

        return seed_nodes, output_nodes, blocks

    def sample_neighbors(self, g, seed_nodes):
        seed_nodes = dgl.utils.prepare_tensor(g, seed_nodes, 'seed_nodes')  #将seed_nodes转变成一个适合图的张量
        self.restart_prob = F1.copy_to(self.restart_prob, F1.context(seed_nodes))  #将restart_prob放在和seed_nodes相同设备上（如GPU）方便计算
        seed_nodes = F1.repeat(seed_nodes, self.num_random_walks, 0) 
        #就是我要从seed_nodes出发进行num_random_walks次的随机游走，那么就要先让seed_nodes张量在第一维重复多次，第一维保存的就是节点个数了，第二维度保存的就是节点的特征
        paths, eids, types = random_walk(
            g, seed_nodes, length=self.random_walk_length, restart_prob=self.restart_prob, return_eids=True)
        
        src_1 = F1.reshape(paths[:, self.random_walk_length-1::self.random_walk_length], (-1,))
       
        src_2 = F1.reshape(paths[:, self.random_walk_length::self.random_walk_length], (-1,)) #将形状变成一维张量

        src = torch.cat([src_1, src_2], dim=0)  #将两种源节点结合
        dst = torch.cat([seed_nodes, seed_nodes], dim=0) #目标节点就是seed_nodes,但是需要接受两种邻居的聚合

        mask_1 = (src_1 != -1)
        mask_2 = (src_2 != -1)
        mask = (src != -1)
        src = src[mask] #掩码
        dst = dst[mask]
        eids = eids[mask_2]

        neighbor_graph = dgl.graph((src, dst))
        etypes = g.edata[dgl.ETYPE][eids]  #得到随机游走经过的边类型赋值给etypes，那etypes就是一个二维张量，第一维度是变得个数，第二维度是边的类型
        etime = g.edata['time'][eids]
        event = g.edata['event'][eids]
        
        neighbor_graph.edata['first_etype'] = torch.cat([etypes[:, 0],etypes[:,0]],dim=0)
        neighbor_graph.edata['second_etype'] = torch.cat([etypes[:, 1],etypes[:,1]],dim=0)

        neighbor_graph.edata['etype'] = torch.cat([etypes[:, 0], etypes[:, 1]], dim=0)
        neighbor_graph.edata['etime'] = torch.cat([etime[:, 0], etime[:, 1]], dim=0)
        neighbor_graph.edata['event'] = torch.cat([event[:, 0], event[:, 1]], dim=0)
        
        neighbor_graph.edata['hete'] = torch.cat([torch.ones(etypes.shape[0]), torch.zeros(etypes.shape[0])])
        neighbor_graph.edata['homo'] = torch.cat([torch.zeros(etypes.shape[0]), torch.ones(etypes.shape[0])])
        neighbor_graph = dgl.to_simple(neighbor_graph, return_counts='count', copy_edata=True)

        return neighbor_graph 

def refine_time(data):
    data = data.sort_values(['time'],kind = 'mergesort')
    return data

def gen_t_labels(data):
    filter = {}
    for row in data.itertuples():
        filter[getattr(row, 'item_id')] = getattr(row, 't_label')
    return filter

#生成图
def generate_graph(data):
    data = data.groupby('user_id').apply(refine_time).reset_index(drop=True) #对每一个用户id的交互情况排序

    user = data['user_id'].values
    item = data['item_id'].values
    time = data['time'].values
    event = data['event'].values
    event_dic = {'post':1,'retweet':2}
    event_map = [event_dic[i] for i in event]
    graph_data = {('item','by','user'):(torch.tensor(item), torch.tensor(user)),
                  ('user','pby','item'):(torch.tensor(user), torch.tensor(item))}
    graph = dgl.heterograph(graph_data) #异构图
    graph.edges['by'].data['time'] = torch.LongTensor(time) #在边属性中添加一个时间的见
    graph.edges['pby'].data['time'] = torch.LongTensor(time) #用户为目标节点的边也一样
    graph.edges['by'].data['event'] = torch.LongTensor(event_map) #在边属性中添加一个时间的见
    graph.edges['pby'].data['event'] = torch.LongTensor(event_map)

    filter = gen_t_labels(data)
    graph.nodes['user'].data['user_id'] = torch.LongTensor(np.unique(user)) #用户节点属性  user_id
    graph.nodes['item'].data['item_id'] = torch.LongTensor(np.unique(item))  #项目节点属性
    graph.nodes['item'].data['t_label'] = torch.LongTensor([filter[id] for id in np.unique(item)])  #文本标签{0，1}

    return graph

#生成用户子图
def generate_user(user, data, graph, item_max_length, user_max_length, train_path, test_path,val_path=None):
    data_user = data[data['user_id'] == user].sort_values('time')  #针对某一个用户的时间戳升序排列
    u_time = data_user['time'].values #用户时间
    u_seq = data_user['item_id'].values #用户所连接的项目序列（item）
    u_labels = data_user['t_label'].values
    split_point = len(u_seq) - 1  #拆分点（就是最后一个时间点前面）
    train_num = 0
    test_num = 0
    # 生成训练数据
    if len(u_seq) < 3:
        return 0, 0 #如果用户序列小于3 则不要这个用户信息了
    else:#用户有足够多的邻居
        test_t = random.sample(list(u_time), 1)[0]
        for j, t  in enumerate(u_time[0:-1]): #对于每一个时间戳（已经排好序了） 取历史时间戳
            if j == 0:  #j的取值就是user邻居节点的个数
                continue
            if j < item_max_length:     #对于历史节点每一个序列生成多个训练数据{i1,i2,i3}->i4,{i1,i2,i3,i4}->i5
                start_t = u_time[0] #起始时间点
            else:  #邻居节点个数大于最大采样个数50，采用重复采样？  item_max_lenth = 50
                start_t = u_time[j - item_max_length]
            # test_j = random.randint(1,split_point)
            # if u_time[j] == u_time[j+1] and j != test_j:
            #     continue
            if u_time[j] == u_time[j+1] and j+1 != len(u_time[0:-1]) -1:
                continue
            sub_u_eid = (graph.edges['by'].data['time'] < u_time[j+1]) & (graph.edges['by'].data['time'] >= start_t)
            sub_i_eid = (graph.edges['pby'].data['time'] < u_time[j+1]) & (graph.edges['pby'].data['time'] >= start_t)
            sub_graph = dgl.edge_subgraph(graph, edges = {'by':sub_u_eid, 'pby':sub_i_eid}, relabel_nodes=False)
            homo_g = dgl.to_homogeneous(sub_graph,edata=['time','event'],return_count=True)[0]

            u_temp = torch.tensor([user])   #锚节点
            his_user = torch.tensor([user])  #采样节点c
            num_texts = len(data)
            
            sample = NeighborSampler(random_walk_length=2, num_random_walks=10, num_neighbors=3, num_layers=2)
            seed_nodes, output_nodes, blocks_1 = sample.sample_blocks(g=homo_g, seed_nodes=u_temp)
            
            
                        
            u_t_labels = u_labels[:j]
            u_t_ratio = float(sum(u_t_labels)) / float(len(u_t_labels))
            if u_t_ratio > 0.66:
                target = 1
            elif u_t_ratio < 0.33:
                target = 0
            else:
                target = 2
            # target = 1 if u_t_ratio > 0.66 else u_t_ratio < 0.33 0 else 3
            # target = u_seq[j+1]    #target 每一个序列的监督标签
            #last_item可以不需要
            last_item = u_seq[j]   #最后一个时间点的文章
            
            data = {
            'blocks': blocks_1,
            'user': torch.tensor([user]),
            'target': torch.tensor([target])}
            
            # 分别计算user和last_item在fin_graph中的索引 
            if j <= split_point and u_time[j] != test_t:
                file_path = train_path+ '/' + str(user) + '/'+ str(user) + '_' + str(j) + '.pickle'
                if os.path.exists(file_path):
                    continue
                else:
                    
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # 创建目录
                    with open(train_path+ '/' + str(user) + '/'+ str(user) + '_' + str(j) + '.pickle', 'wb') as f:
                        pickle.dump(data, f)
                # save_graphs(train_path+ '/' + str(user) + '/'+ str(user) + '_' + str(j) + '.bin',blocks_1 ,
                #             {'user': torch.tensor([user]), 'target': torch.tensor([target])})
                train_num += 1
            if j == split_point - 1:
                file_path = val_path + '/' + str(user) + '/' + str(user) + '_' + str(j) + '.pickle'
                if os.path.exists(file_path):
                    continue
                else:
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # 创建目录
                    with open(val_path + '/' + str(user) + '/' + str(user) + '_' + str(j) + '.pickle', 'wb') as f:
                        pickle.dump(data, f)
                    # save_graphs(val_path + '/' + str(user) + '/' + str(user) + '_' + str(j) + '.bin', blocks_1,
                #             {'user': torch.tensor([user]), 'target': torch.tensor([target])})
            if u_time[j] == test_t:
                file_path = test_path + '/' + str(user) + '/' + str(user) + '_' + str(j) + '.pickle'
                if os.path.exists(file_path):
                    continue
                else:
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # 创建目录
                    with open(test_path + '/' + str(user) + '/' + str(user) + '_' + str(j) + '.pickle', 'wb') as f:
                        pickle.dump(data, f)
                # save_graphs(test_path + '/' + str(user) + '/' + str(user) + '_' + str(j) + '.bin', blocks_1,
                #             {'user': torch.tensor([user]), 'target': torch.tensor([target])})
                test_num += 1
        return train_num, test_num


def generate_data(data, graph, item_max_length, user_max_length, train_path, test_path, val_path, job=10):
    user = data['user_id'].unique()
    a = Parallel(n_jobs=job)(delayed(lambda u: generate_user(u, data, graph, item_max_length, user_max_length, train_path, test_path, val_path))(u) for u in user)
    return a
    #parallel  并行计算

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='sample', help='data name: sample')
    parser.add_argument('--graph', action='store_true', help='no_batch')
    parser.add_argument('--item_max_length', type=int, default=50, help='most recent')
    parser.add_argument('--user_max_length', type=int, default=50, help='most recent')
    parser.add_argument('--job', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--k_hop', type=int, default=2, help='k_hop')
    opt = parser.parse_args()
    data_path = '/home/kayzhou/yuexutong/DGSR-master/DGCR-Dubleh/stance_data_0505.csv'
    graph_path = '/home/kayzhou/yuexutong/DGSR-master/DGCR-Dubleh/data/network/Doubleh-DGCR_hete_g_0510.pickle'
    data = pd.read_csv(data_path).groupby('user_id').apply(refine_time).reset_index(drop=True)
    data['time'] = data['time'].astype('int64')
    # if opt.graph:
    #     graph = generate_graph(data)
    #     save_graphs(graph_path, graph)
    # else:
    if not os.path.exists(graph_path):
        graph = generate_graph(data)
        save_graphs(graph_path, graph)  #这里生成的异构图是一个字典存储的  （user，by，item）：tensor（），tensor（）两个张量中存储user和item的索引  在datafrmae中是一行  表示一一对应
    else:
        graph = dgl.load_graphs(graph_path)[0][0]
    train_path = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/train'
    val_path = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/val'
    test_path = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/test'
    #generate_user(41, data, graph, opt.item_max_length, opt.user_max_length, train_path, test_path, k_hop=opt.k_hop)
    print('start:', datetime.datetime.now())
    all_num = generate_data(data, graph, opt.item_max_length, opt.user_max_length, train_path, test_path, val_path, job=opt.job)
    train_num = 0
    test_num = 0
    for num_ in all_num:
        train_num += num_[0]
        test_num += num_[1]
    print('The number of train set:', train_num)
    print('The number of test set:', test_num)
    print('end:', datetime.datetime.now())


