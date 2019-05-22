import unicodedata
import string
import re

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

SOS_token = 0
EOS_token = 1

class attention_decoder_rnn(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(attention_decoder_rnn, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attention = nn.Linear(2 * hidden_size, MAX_LENGTH)
        self.attention_combined = nn.Linear(2 * hidden_size, MAX_LENGTH)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
    def forward(self, input, hidden, encoder_outputs):
        embedded = self.Embedding(input).view(1,1,-1)
        attention_weights = F.softmax(self.attention(torch.cat((embedded[0], hidden[0]),1)), dim = 1)
        attention_applied = torch.bmm(attention_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attention_applied[0]), 1)
        output = self.attention_combined(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
