
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torchsummary import summary

class LSTM_Classifier(torch.nn.Module):
    def __init__(self,n_features,seq_length, n_layers, nb_classes):
        super(LSTM_Classifier, self).__init__()
        self.n_features = n_features # nb of features (e.g. R, G, B if sequence on pixels is consideried)
        self.seq_len = seq_length # Length of the sequence
        self.n_hidden = 20 # number of hidden states
        self.n_layers = n_layers # number of LSTM layers (stacked)

        self.l_lstm = torch.nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True)
        # according to pytorch docs LSTM output is 
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Linear(self.n_hidden*self.seq_len, nb_classes)

        

    def init_hidden(self, batch_size, device):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden).to(device)
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden).to(device)
        self.hidden = (hidden_state, cell_state)


    def forward(self, x):        
        batch_size, seq_len, _ = x.size()
        
        lstm_out, self.hidden = self.l_lstm(x, self.hidden)
        # lstm_out(with batch_first = True) is 
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest       
        # .contiguous() -> solves tensor compatibility error
        x = lstm_out.contiguous().view(batch_size,-1)
        #print(x.shape)
        return self.l_linear(x)

if __name__ == "__main__":
    net = LSTM_Classifier(n_features=1, seq_length=224, n_layers=1, nb_classes=2)

    print(net)