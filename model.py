import torch.nn.functional as F
from torch import nn
import torch

class MLP(torch.nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout_prob=0.5):
        super(MLP, self).__init__()
        self.d_in = d_in
        self.linear1 = torch.nn.Linear(d_in, d_hidden)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.linear2 = torch.nn.Linear(d_hidden, d_out)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)

        

    def forward(self, X):
        X = X.view(-1, self.d_in)
        X = self.relu1(self.linear1(X))
        X = self.dropout1(X)
        X = self.relu2(self.linear2(X))
        X = self.dropout2(X)
        return F.softmax(X, dim=1)