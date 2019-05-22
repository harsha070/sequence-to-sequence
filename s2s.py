import unicodedata
import string
import re

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

SOS_token = 0
EOS_token = 1

def unicode_to_ascii(s):
    return str(unicodedata.normalize('NFKD',s).encode('ascii', 'ignore'))

def normalize_string(s):
    s = unicode_to_ascii(s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

class language:
    def __init__(self, name):
        self.name = name
        self.word_to_id = {}
        self.id_to_word = {}
        self.word_count = {}
        self.n_words = 2
        self.word_to_id['SOS'] = 0
        self.word_to_id['EOS'] = 1
        self.id_to_word[0] = 'SOS'
        self.id_to_word[1] = 'EOS'
    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)
    def add_word(self, word):
        if word in self.word_to_id:
            self.word_count[word] += 1
        else:
            self.word_to_id[word] = self.n_words
            self.id_to_word[self.n_words] = word
            self.n_words += 1
            self.word_count[word] = 1

english = language('english')
french = language('french')

corpus = open('data/data/eng-fra.txt', encoding = 'utf-8').read().strip().split('\n')

pairs = [[normalize_string(s) for s in line.split('\t')] for line in corpus]

MAX_LENGTH = 10

print(pairs[2])

pairs = [pair for pair in pairs if (len(pair[0].split(' '))<MAX_LENGTH and len(pair[1].split(' '))<MAX_LENGTH)]

for pair in pairs:
    english.add_sentence(pair[0])
    french.add_sentence(pair[1])

print(len(pairs))
print(english.n_words)
print(french.n_words)

hidden_size = 128

encoder = encoder_rnn(english.n_words, hidden_size)


learning_rate = 0.01

encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr = learning_rate)
decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr = learning_rate)
criterion = nn.NLLLoss()

def convert_input_sentence_to_tensor(lang, sentence):
    indices = [lang.word_to_id(word) for word in sentence]
    indices.append(EOS_token)
    return torch.Tensor(indices).view(-1,1)

def convert_output_sentence_to_tensor(lang, sentence):
    indices = [EOS_token]
    indices = indices + [lang.word_to_id(word) for word in sentence]
    return torch.Tensor(indices).view(-1,1)

def training():
    for pair in pairs:
        input_sentence = pair[0].split(' ')
        output_sentence = pair[1].split(' ')

        input_tensor = convert_input_sentence_to_tensor(english, input_sentence)
        output_tensor = convert_input_sentence_to_tensor(french, output_sentence)

        input_length = len(input_sentence)
        output_length = len(output_sentence)

        encoder_hidden = encoder.initialize_hidden()
        encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size)

        for i in range(input_length):
            encoder_output, encoder_hidden = encoder.forward(input_tensor[i], encoder_hidden)
            encoder_outputs[i] = encoder_output[0,0]

        decoder_hidden = encoder_hidden
        decoder_input = SOS_token

        loss = 0

        for i in range(output_length):
            decoder_output, decoder_hidden, decoder_attention = decoder.forward(decoder_input, decoder_hidden, encoder_outputs)
            value, index = decoder_output.topk(1)
            decoder_input = index.squeeze().detach()

            loss += criterion(decoder_output, output_tensor[i])

            if(decoder_input.item() == EOS_token):
                break
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
