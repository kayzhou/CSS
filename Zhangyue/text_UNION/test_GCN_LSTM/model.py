import pickle
import numpy as np
import json
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from HLSTM import HLSTM
from tqdm import tqdm
import gc


def generate_examples_hlstm(text_inputs_raw, word_to_idx):
    all_text_input, all_text_lengths = [], []
    doc_length = []
    for doc in text_inputs_raw:
        j = 0
        text_input=[]
        for word in doc:
            if word in word_to_idx:
                text_input.append(word_to_idx[word])
                j=j+1
                if j>=64:
                    break
        
        text_lengths = len(text_input)
        if text_lengths == 0:
            text_input = np.zeros([64,200])
            text_lengths = len(text_input)
        

        all_text_input.append(text_input)
        all_text_lengths.append(text_lengths)

    all_text_lengths = np.array(all_text_lengths)

    
    part = all_text_lengths
    print(len(part), max(part), sum(part <= 32), sum(part <= 64))
    return all_text_input, all_text_lengths


def process_batch(all_text_input, all_text_lengths,
    batch_index, vocab_size, perm=None):
    
    MAX_WORD_LEN = 64
    if (perm is not None):
        batch_index = [perm[i] for i in batch_index]

    batch_input_raw = [all_text_input[i] for i in batch_index]
    batch_lengths_raw = [all_text_lengths[i] for i in batch_index]
    batch_doc_lengths = batch_lengths_raw
    #batch_doc_lengths = [len(lengths) for lengths in batch_lengths_raw]
    max_doc_length = min(MAX_WORD_LEN, max(batch_lengths_raw))
    for i in range(len(batch_lengths_raw)):
        if (batch_doc_lengths[i] > max_doc_length):
            batch_doc_lengths[i] = max_doc_length
    # print(batch_sent_lengths, max_sent_length)

    batch_input = []
    for doc_input, sent_length in zip(batch_input_raw, batch_lengths_raw):
        if (sent_length >= max_doc_length):
            batch_input.append(doc_input[:max_doc_length])
        else: 
            batch_input.append(np.vstack((doc_input, np.array([[0]*200]*(max_doc_length - sent_length)))))
    
    batch_input = np.asarray(batch_input)
    batch_doc_lengths = np.asarray(batch_doc_lengths)
    
    return batch_input, batch_doc_lengths


class GCNModel(nn.Module):
    def __init__(self, data, num_features, num_nodes, hidden, support, num_bases, num_classes, 
        text_model='HLSTM', bias_feature=False):
        super(GCNModel, self).__init__()

        self.input_layer = InputLayer(data, num_features, text_model, bias_feature)
        self.gc1 = GraphConvolution(num_nodes, hidden, support, num_bases=num_bases,
            activation='tanh')
        self.gc2 = GraphConvolution(hidden, num_features, support, num_bases=num_bases,
                    activation='tanh')
        self.clf_bias = nn.Linear(num_features, num_classes)
       

class InputLayer(nn.Module):
    def __init__(self, data, num_feature, text_model='HLSTM', bias_feature=False):

        super(InputLayer, self).__init__()

        num_nodes = data['num_nodes']  
        num_docs = data['num_docs']  
        num_non_docs= data['num_non_docs']
        
        # Not used, but kept to reproduce result
        #self.node_embedding = nn.Embedding(num_nodes, num_feature)
        self.node_embedding = nn.Embedding(num_nodes, num_feature).cuda()
        self.node_embedding.weight.requires_grad=False
        self.text_model = text_model
        self.bias_feature = bias_feature

        #if (bias_feature):
         #   fin = open('data/news_article_feature.pickle', 'rb')
          #  feature_matrix = pickle.load(fin)
           # self.feature_matrix = torch.FloatTensor(feature_matrix)
            #self.feature_matrix = torch.FloatTensor(feature_matrix).cuda()
            #fin.close()


        # tokenized_docs.json contains the tokenized content of news articles
        #存成一个字典格式，键为id，值为词构成的列表
        # with open(r'/home/kayzhou/zhangyue/text/data_GCN/data_processed/tweet_id_token_10.json') as fin:
        #     doc_inputs_raw = json.load(fin)
        #     fin.close()
        #读取方式需要改 2.15号
        doc_inputs_raw={}
        with open(r'/home/kayzhou/zhangyue/text/data_GCN/data_processed/tweet_id_token_10.txt',encoding="utf-8") as file:
            for line_ in tqdm(file):
                line_=line_.strip().split(" ")
                if len(line_)==1:
                    doc_inputs_raw[line_[0]] = []
                else:
                    emb = line_[1:]
                    doc_inputs_raw[line_[0]] = emb
            


        text_inputs_raw = []
        for doc_id in range(num_non_docs, num_nodes):
            text_inputs_raw.append(doc_inputs_raw[str(doc_id)])
        
        del doc_inputs_raw
        
        fin = open(r'/home/kayzhou/zhangyue/text/data_ML_ANNs/glove.twitter.27B.200d.txt',encoding="utf-8")
        word_to_idx={}
        for line_ in tqdm(fin.readlines()):
            line_=line_.strip().split(" ")
            emb = np.asarray([float(i) for i in line_[1:201]])
            word_to_idx[line_[0]] = emb       
                

        # print(len(word_to_idx), word_embeddings.shape)

        all_text_input, all_text_lengths = generate_examples_hlstm(text_inputs_raw, word_to_idx)
        HIDDEN_SIZE = 128 #基线模型为128,默认代码为64
        BIAS_FEAT_SIZE = 141
        self.doc_hlstm = HLSTM(HIDDEN_SIZE, word_to_idx)
        # if (bias_feature):
        #     #self.linear = nn.Linear(HIDDEN_SIZE+BIAS_FEAT_SIZE, num_feature).cuda()
        #     self.linear = nn.Linear(HIDDEN_SIZE+BIAS_FEAT_SIZE, num_feature)
        # else:
        #   #双向的lstm hidden_size*2
        #     #self.linear = nn.Linear(HIDDEN_SIZE*2, num_feature).cuda()
        #     self.linear = nn.Linear(HIDDEN_SIZE*2, num_feature)
        #     #self.linear = nn.Linear(HIDDEN_SIZE, num_feature).cuda()
        
        #双向LSTM用*2，单向不用
        #self.linear = nn.Linear(HIDDEN_SIZE*2, num_feature)
        self.linear = nn.Linear(HIDDEN_SIZE, num_feature)
        
        self.all_text_input, self.all_text_lengths = all_text_input, all_text_lengths

    def get_doc_embed(self, idx): 
        # if (self.bias_feature):
        #     bias_feat = self.feature_matrix.index_select(0, idx)

        #     print('\t\t --- doc embedding used')
        # 4000有什么用
        batch_input,  batch_doc_lengths = process_batch(
            self.all_text_input, self.all_text_lengths, idx, 400000)
        text_embeds = self.doc_hlstm(batch_input, batch_doc_lengths)
        # if (self.bias_feature):
        #    text_embeds = torch.cat((text_embeds, bias_feat), 1) 
        text_embeds = self.linear(text_embeds)
        return text_embeds


# This implementation is based on the code at https://github.com/tkipf/relational-gcn/blob/master/rgcn/layers/graph.py
class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, support=1, 
                 activation='linear', num_bases=-1, bias=False):
        
        super(GraphConvolution, self).__init__()

        if (activation == 'linear'):
            self.activation = None
        elif (activation == 'sigmoid'):
            self.activation = nn.Sigmoid()
        elif (activation == 'tanh'):
            self.activation = nn.Tanh()
        else:
            print('Error: activation function not available')
            exit()

        self.input_dim = input_dim
        self.output_dim = output_dim  # number of features per node
        self.support = support  # filter support / number of weights

        assert support >= 1

        self.bias = bias
        self.num_bases = num_bases

        if self.num_bases > 0:
            self.W = [Parameter(nn.init.xavier_uniform_(
                torch.FloatTensor(self.input_dim, self.output_dim)))
                for i in range(self.num_bases)]
            self.W_comp = Parameter(nn.init.xavier_uniform_(
                torch.FloatTensor(self.support, self.num_bases)))
        else:
            self.W = [Parameter(nn.init.xavier_uniform_(
                torch.FloatTensor(self.input_dim, self.output_dim)))
                for i in range(self.support)]
        for idx, item in enumerate(self.W):
            self.register_parameter('W_%d' % idx, item)

        if self.bias:
            self.b = Parameter(torch.FloatTensor(self.output_dim, 1))
        
    def forward(self, inputs, mask=None):
        features = inputs[0]
        A = inputs[1:]  # list of basis functions

        # convolve
        supports = []
        if self.num_bases > 0:
            V = []
            for i in range(self.support):
                basis_weights = []
                for j, a_weight in enumerate(self.W):
                    basis_weights.append(self.W_comp[i][j] * a_weight)
                V.append(torch.stack(basis_weights, dim=0).sum(0))
            for i in range(self.support):
                # print(V[i].size())
                supports.append(torch.spmm(features, V[i]))
        else:
            for a_weight in self.W:
                supports.append(torch.spmm(features, a_weight))

        outputs = []
        for i in range(self.support):
            # print(features.size(), A[i].size())
            outputs.append(torch.spmm(A[i], supports[i]))

        output = torch.stack(outputs, dim=1).sum(1)            

        if self.bias:
            output += self.b
        
        if self.activation is not None:
            return self.activation(output)
        else:
            return output
