import torch
import torch.nn as nn
import torch.nn.functional as F
from helper_functions import *

class EncoderRNN(nn.Module):
    def __init__(self, cell_type, input_size, embedding_size, hidden_size, num_layers, bidirectional, dropout, device):
        super(EncoderRNN, self).__init__()
        CELL = get_cell_type(cell_type)
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True if bidirectional == 'True' else False
        self.bidirectional_size = 2 if self.bidirectional else 1
        self.dropout = dropout
        self.device = device

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.cell = CELL(input_size=embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout, bidirectional=self.bidirectional)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.cell(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.bidirectional_size, 1, self.hidden_size, device=self.device)
    
class DecoderRNN(nn.Module):
    def __init__(self, cell_type, output_size, embedding_size, hidden_size, num_layers, bidirectional, dropout, device):
        super(DecoderRNN, self).__init__()
        CELL = get_cell_type(cell_type)
        self.input_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True if bidirectional == 'True' else False
        self.bidirectional_size = 2 if self.bidirectional else 1
        self.dropout = dropout
        self.device = device

        self.embedding = nn.Embedding(output_size, self.embedding_size)
        self.cell = CELL(input_size=embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout, bidirectional=self.bidirectional)
        self.out = nn.Linear(self.bidirectional_size*hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.device = device

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.cell(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.bidirectional_size, 1, self.hidden_size, device=self.device)