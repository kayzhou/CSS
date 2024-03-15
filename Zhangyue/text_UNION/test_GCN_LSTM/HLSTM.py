import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence 

class HLSTM(torch.nn.Module):
    def __init__(self, hidden_size, word_to_idx):
        """
        In the constructor we construct instances that we will use
        in the forward pass.
        """
        super(HLSTM, self).__init__()
        

        #self.sent_LSTM = nn.LSTM(embedding_size, hidden_size, 1, batch_first=True).cuda()
        self.doc_LSTM = nn.LSTM(200, hidden_size, 2, bidirectional=False, batch_first=True).cuda()
        #self.doc_LSTM = nn.LSTM(200, hidden_size, 2, bidirectional=True, batch_first=True)
        #self.doc_LSTM = nn.LSTM(200, hidden_size, 1, bidirectional=True，batch_first=True).cuda()
        #self.doc_LSTM = nn.LSTM(200, hidden_size, 1, batch_first=True).cuda()
        self.word_to_idx = word_to_idx



    def forward(self, batch_input, document_lengths):
        
    
        batch_input_var = torch.tensor(batch_input, dtype=torch.float).cuda()
        #batch_input_var = torch.tensor(batch_input, dtype=torch.float)
        
        doc_out_packed, _ = self.doc_LSTM(batch_input_var)
        doc_out_packed = doc_out_packed[:, -1, :]
        

        return doc_out_packed

    def get_scores(self, batch_input,document_lengths, label_embedding):
        doc_embeds = self.forward(batch_input, document_lengths)
        scores = label_embedding(doc_embeds)
        # scores = torch.sigmoid(self.W_reg(doc_embeds)) * 4
        
        return scores

    def predict(self, batch_input,  document_lengths, label_embedding):
        scores = self.get_scores(batch_input, document_lengths, label_embedding)
        # print(scores)

        # pred_label = scores.view(-1)
        _, pred_label = torch.max(scores, 1)
        pred_label = to_value(pred_label)
        
        return pred_label

        

        


