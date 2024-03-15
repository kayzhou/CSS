import datetime
import torch
from sys import exit
import pandas as pd
import numpy as np
from DGSR import DGSR, collate, collate_test
from dgl import load_graphs
import pickle
from utils import myFloder
import warnings
import argparse
import os
import sys
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from DGSR_utils import eval_metric, mkdir_if_not_exist, Logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sentence_transformers import SentenceTransformer
from sklearn import metrics


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='sample', help='data name: sample')
parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=50, help='hidden state size')
parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--l2', type=float, default=0.0001, help='l2 penalty')
parser.add_argument('--feat_drop', type=float, default=0.3, help='drop_out')
parser.add_argument('--attn_drop', type=float, default=0.3, help='drop_out')
parser.add_argument('--layer_num', type=int, default=3, help='GNN layer')
parser.add_argument('--item_max_length', type=int, default=50, help='the max length of item sequence')
parser.add_argument('--user_max_length', type=int, default=50, help='the max length of use sequence')
parser.add_argument('--k_hop', type=int, default=3, help='sub-graph size')  #这里原本是2
parser.add_argument('--gpu', default='4')
parser.add_argument('--last_item', action='store_true', help='aggreate last item')
parser.add_argument("--record", action='store_true', default=False, help='record experimental results')
parser.add_argument("--val", action='store_true', default=False)
parser.add_argument("--model_record", action='store_true', default=False, help='record model')
parser.add_argument("--save_path",type=str,default='model/GRSN')

#注意 action就相当于一个开关 action=‘store_true’的时候  在调用时候只需要 python main.py --record 就直接将其设为true  否则就是默认值false
#好处就是不用再调用参数的时候设置参数的值了
opt = parser.parse_args()
args, extras = parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
device = torch.device('cuda:0')
print(opt)

# loading data
data = pd.read_csv('./Data/' + opt.data + '.csv')
user = data['user_id'].unique() #unique()对series去重 返回一个numpy数组
item = data['item_id'].unique()
user_num = len(user) #用户节点
item_num = len(item)#项目节点 
test_root = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/test/'
test_set = myFloder(test_root, load_graphs)


# print('train number:', train_set.size)   #一个train_batch就是一个子图
print('test number:', test_set.size)
print('user number:', user_num)
print('item number:', item_num)


def evaluate(model,dataloader, criterion):
    model.eval()
    total_loss = 0.
    all_labels, all_preds = [], []
    for user, batch_graph, label in dataloader:
        with torch.no_grad():
            logits = model(batch_graph.to(device), user.to(device))
            loss = criterion(logits, label.to(device))
            true_labels = label.data.cpu()
            all_labels += true_labels.tolist()
            # all_preds += logits.argmax(1).tolist()
            all_preds += torch.max(logits.data,1)[1].cpu().tolist()
            # pred = torch.max(logits.data,1)[1].cpu()
            total_loss += loss.item()

    total_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    # precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro')
    report = metrics.classification_report(all_labels, all_preds, target_names=["Trump","Biden","NS"], digits=4)
    confusion = metrics.confusion_matrix(all_labels, all_preds)
    print(confusion)
    return accuracy, total_loss, report, confusion



test_data = DataLoader(dataset=test_set, batch_size=opt.batchSize, collate_fn=collate, pin_memory=True, num_workers=0) #原来进程数是8



# 初始化模型
model = DGSR(user_num=user_num, item_num=item_num, input_dim=opt.hidden_size, item_max_length=opt.item_max_length,
             user_max_length=opt.user_max_length, feat_drop=opt.feat_drop, attn_drop=opt.attn_drop,
             last_item=opt.last_item,layer_num=opt.layer_num).cuda()
optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
loss_func = nn.CrossEntropyLoss()
# best_result = [0, 0, 0, 0, 0, 0]   # hit5,hit10,hit20,mrr5,mrr10,mrr20  #知识图谱嵌入得性能评估
# best_epoch = [0, 0, 0, 0, 0, 0]
# stop_num = 0

#test
print('start presicion')
model.load_state_dict(torch.load('DGSR_model'))
model.eval()
start_time = datetime.datetime.now()
test_acc, test_loss, report,confusion = evaluate(model,test_data,loss_func)
# print("Test_loss {:.4f}, Test_acc {:.2f}".format(test_loss,test_acc))
msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
print(msg.format(test_loss, test_acc))
print("Precision, Recall and F1-Score...")
print(report)
print("Confusion Matrix...")
print(confusion)

