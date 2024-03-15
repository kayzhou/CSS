import scipy.sparse as sp
import torch
import numpy as np
import pickle as pkl

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    #将scipy稀疏矩阵转换为Torch稀疏张量。
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    shape = torch.Size(sparse_mx.shape)
    indices_array = np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    # print(indices_array.shape, type(indices_array))
    
    if (indices_array.shape[1] > 0):
        indices = torch.LongTensor(indices_array)
        values = torch.FloatTensor(sparse_mx.data)
        sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    else:
        sparse_tensor = torch.sparse.FloatTensor(shape[0], shape[1])
    return sparse_tensor


def load_data(args, dirname, use_cuda, SUPERVISE_FLAG):

    # Load Data
    DATASET = args['dataset']
    with open(dirname  + DATASET + '.pkl', 'rb') as f:
        raw_data = pkl.load(f)
    #print(raw_data)
    
    A = [raw_data['A'][0].tolil()]

    y = np.array(raw_data['y'])
  
    
    idx_train = raw_data['train_idx']

    idx_valid = raw_data['valid_idx']
    idx_test = raw_data['test_idx'] 
    print(len(idx_train),len(idx_valid),len(idx_test))
    
    #all_label为类别的集合，这里为[0，1，2]
    all_labels = raw_data['all_labels'] 
    
    all_users = raw_data['all_user'] 
    all_docs = raw_data['all_docs']
    
    num_docs = len(all_docs)
    
    labels = y
    
    print("y:"+str(len(y)))
    
    
   
    num_nodes = A[0].shape[0]
    print("num_nodes:"+str(num_nodes))
    num_non_docs = num_nodes - num_docs
    support = len(A)
    print("support:"+str(support))
    data = {}
    
    idx_train_set, idx_test_set = set(idx_train), set(idx_test)
    
    # # test with reduced network
    # with open(dirname + '/newsbias_random_1_untyped_40.pickle', 'rb') as f:
    #     data_reduced = pkl.load(f)
    # A = data_reduced['A']

    # only use the labels of political users at training time in the distant supervision case ('unsup1')

    
    # print(len(A), len(all_labels), len(all_poli_users), len(all_share_users), len(all_docs))
    # print(len(idx_train), len(idx_valid), len(idx_test))
    
    # Define one-hot dummy feature matrix 
    X = sp.eye(num_nodes).tocsr() 
    # Normalize adjacency matrices individually
    for i in range(len(A)):
        d = np.array(A[i].sum(1)).flatten()
        d_inv = 1. / d
        d_inv[np.isinf(d_inv)] = 0.
        D_inv = sp.diags(d_inv)
        A[i] = D_inv.dot(A[i]).tocsr()
        
    inputs = [sparse_mx_to_torch_sparse_tensor(item) for item in [X] + A]
    # print(len(inputs))
    # for input in inputs:
    #     print(input.size())
    labels_train = torch.LongTensor(labels[idx_train])
    labels_valid = torch.LongTensor(labels[idx_valid])
    labels_test = torch.LongTensor(labels[idx_test])
        
    if (use_cuda):
        inputs = [item.cuda() for item in inputs]
        labels_train = labels_train.cuda()
        labels_valid = labels_valid.cuda()
        labels_test = labels_test.cuda()
    #A[0].tocoo().row不为0的元素所在的行索引，A[0].tocoo().col不为0元素所在的列索引
    x_pos, y_pos = A[0].tocoo().row, A[0].tocoo().col
    #print("x_pos,y_pos:"+str(x_pos)+" "+str(y_pos))
    all_docs_set, all_set = set(all_docs), set(all_users)
    
    node2adj = {}
    for xi, yi in zip(x_pos, y_pos):
        if (xi not in all_docs_set or yi not in all_set):
            continue
        if xi not in node2adj:
            node2adj[xi] = []
        node2adj[xi].append(yi)
    #share to doc or pol
    
    num_edges_list, num_labels_list = [], [0, 0, 0]   
    #num_edges_list统计测试集中的节点存在几个邻居节点
    #num_labels_list统计测试集中各个类别的节点有几个
    for node in idx_test:
        if node not in node2adj:
            num_edges_list.append(0)
        else:
            num_edges_list.append(len(node2adj[node]))
        num_labels_list[labels[node]] += 1
    # print(len(num_edges_list), sum(num_edges_list)/len(num_edges_list))
    # print(sum(num_labels_list), np.array(num_labels_list)/sum(num_labels_list))
    # print(len(node2adj))

    data['idx_train'] = idx_train
    data['idx_valid'] = idx_valid
    data['idx_test'] = idx_test
    data['idx_train_set'] = idx_train_set 
    data['idx_test_set'] = idx_test_set 
    data['inputs'] = inputs 
    data['labels_train'] = labels_train 
    data['labels_valid'] = labels_valid 
    data['labels_test'] = labels_test
    data['num_nodes'] = num_nodes
    data['num_docs'] = num_docs 
    print(len(data["inputs"]))

    
    #num_non_docs参数意义是什么
    data['num_non_docs'] = num_non_docs
    
    #字典格式，键为节点，值为邻居节点列表
    data['node2adj'] = node2adj
    
    #support参数意义是什么
    data['support'] = support

    return raw_data, data
