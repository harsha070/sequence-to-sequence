import unicodedata
import string
import re

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

SOS_token = 0
EOS_token = 1

class encoder_rnn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(encoder_rnn, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.softmax = nn.LogSoftmax(dim = 1)
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1,1,-1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden
    def initialize_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)
