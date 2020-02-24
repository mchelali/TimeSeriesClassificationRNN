import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torchsummary import summary
from tslearn.datasets import UCR_UEA_datasets


class LSTMTagger(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, nb_classes):
        super(LSTMTagger, self).__init__()


        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_size, nb_classes)


        # Init as zeros the 1st input of the LSTM
        self.h0 = torch.randn(1, 1, hidden_size)
        self.c0 = torch.randn(1, 1, hidden_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x.view(x.size()), (self.h0, self.c0))
        tag_space = self.hidden2tag(lstm_out.view(len(x), -1))
        #tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_space

model = LSTMTagger(input_size=3, hidden_size=512, num_layers=1, nb_classes=2)

a = torch.randn(50, 1, 3)
print(model.lstm(a)[0].shape)
print(model)
#print(summary(model, input_size=(100, 3)))

"""
X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset("TwoPatterns")
print("Train shape", X_train.shape)
print("Test shape", X_test.shape)

"""

