from transformers import get_linear_schedule_with_warmup
import dgl
import torch
import torch.nn as nn
import pickle
import pandas as pd
import datetime
import networkmodel
from utils import parse_args, set_seed, evaluate, myFloder,collate
from torch.utils.data import Dataset, DataLoader


def load_graphs(file_path):
    return pickle.load(open(file_path,'rb'))

def main():
    
    
    args = parse_args()
    if not torch.cuda.is_available():
        args.mode = 'cpu'
    device = torch.device(args.mode)
    set_seed(args.seed) 
    
    hete_g = dgl.load_graphs('hete_dglsave_graph')[0][0]

    homo_g = dgl.to_homogeneous(hete_g, store_type=True)  #吧异构图转换成同构图
    num_users = hete_g.num_nodes('user')
    # num_users = len(users)  #用户节点数量
    num_texts = len(list(range(homo_g.ndata[dgl.NTYPE].shape[0]))) - num_users  #文本节点的数量

    #节点特征
    text_features = pickle.load(open('text_features.pickle', 'rb'))
    text_features = torch.tensor(text_features)
    user_id = homo_g.ndata[dgl.NID][1959698:]
    user_embedding = nn.Embedding(1959698, 768) #用户节点初始化嵌入   nn.Embedding(num_embedding,embedding_dim)  （嵌入字典大小（多少词），每个词词向量维度）
            #初始化embedding词向量，包含user_num个词，每个词都使用hidden_size维度初始化（高斯分布）  在后面引用的是使用索引对
    user_features = user_embedding(user_id)
    features = torch.cat([text_features, user_features], dim=0)
    features = features.to(device)
    

    model = networkmodel.GNet(in_size=user_features.size(1), hid_size=args.hidden_size, out_size=3,
                            num_layers=args.num_layers)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss().to(device)
    
    train_root = 'Newdata/stance_blocknew0731_50_50_3/train/'  #再这个根目录下面存储的是子图采样结果  是一个一个子图文件
    test_root = 'Newdata/stance_blocknew0731_50_50_3/test/'
    val_root = 'Newdata/stance_blocknew0731_50_50_3/val/'
    
    
    train_set = myFloder(train_root, load_graphs,args.jiaohu)
    test_set = myFloder(test_root, load_graphs,1)
    val_set = myFloder(val_root,load_graphs,1)

    train_dataloader = DataLoader(dataset=train_set, batch_size=32, collate_fn=collate, shuffle=False, pin_memory=True)
    valid_dataloader = DataLoader(dataset=val_set, batch_size=32, collate_fn=collate, shuffle=False, pin_memory=True)
    test_dataloader = DataLoader(dataset=test_set, batch_size=32, collate_fn=collate, shuffle=False, pin_memory=True)
    print('train number:', train_set.size)   #一个train_batch就是一个子图
    print('test number:', test_set.size)
    


    num_training_steps = args.num_epochs * len(train_dataloader) #训练迭代次数
    num_warmup_steps = num_training_steps * args.warmup_ratio #预训练迭代次数
    no_decay = ['bias'] 
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], #any（）中的含义：设定的无需更新的参数是否在模型参数中
            'weight_decay': args.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
            'weight_decay': 0.
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )  #预热学习率在预热期间学习率从0（线性）非线性增长到优化器的初始值，之后会从初始值缩减到0

    # with tqdm(total=num_training_steps) as pbar:  #手动添加进程
    dev_best_loss = float('inf')
    for epoch in range(args.num_epochs):
        epoch_loss = 0
        iter = 0
        start = datetime.datetime.now()
        print('start training: ', datetime.datetime.now())
        model.train()
        for i, (input_nodes_i, output_nodes_i, blocks_raw_i,label_i) in enumerate(train_dataloader):

            batch_loss = 0.0
            optimizer.zero_grad()

            for input_nodes, output_nodes, blocks_raw,label in zip(input_nodes_i, output_nodes_i, blocks_raw_i,label_i):
            
                iter += 1
                blocks = [block.to(device) for block in blocks_raw]
                # if model_handler.mode == 'homo':  #同构图的inputnodes和outputsnodes都是所有节点
                feature = features[blocks[0].srcdata[dgl.NID]]   #blocks[0].srcdata[dgl.NID]  就是input_nodes的原始id
                label = label.unsqueeze(0)
                labels = label.to(device)
                # print(labels)

                logits = model(blocks, feature)
                # print(logits)
                loss = criterion(logits, labels)

            
                loss = loss / args.gradient_accumulation_steps
                batch_loss += loss
                loss.backward()
                # print('单个样本用时', datetime.datetime.now())

            if (i + 1) % args.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm) #梯度裁剪
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            
            # batch_loss /= 4
            epoch_loss += batch_loss.item()
            if iter % 10000 == 0:
                print('Iter {}, loss {:.4f}'.format(iter, epoch_loss/iter), datetime.datetime.now())
        print('一轮的时间：',datetime.datetime.now()-start)
        print('开始测试')
        eval_loss, eval_accuracy, eval_f1, eval_precision, eval_recall= \
            evaluate(model, features, valid_dataloader, criterion, device,num_texts,test = False)
        print('保存模型')
        if eval_loss < dev_best_loss:
            torch.save(model.state_dict(),'Model')
            dev_best_loss = eval_loss

        print(f'epoch: {epoch+1:02}')
        print(f'\teval_loss: {eval_loss:.3f} | eval_accuracy: {eval_accuracy*100:.2f}% | eval_f1: {eval_f1*100:.2f}% | eval_precision: {eval_precision*100:.2f}% | eval_recall: {eval_recall*100:.2f}%')
    
    
    print('start presicion')
    model.load_state_dict(torch.load('Model'))
    model.eval()
    print(datetime.datetime.now())
    test_acc, test_loss, report,confusion = evaluate(model,features, test_dataloader, criterion, device,num_texts,test=True)
    # print("Test_loss {:.4f}, Test_acc {:.2f}".format(test_loss,test_acc))
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(report)
    print("Confusion Matrix...")
    print(confusion)

if __name__ == '__main__':
    main()
