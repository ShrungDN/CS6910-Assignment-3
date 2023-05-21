import torch
import torch.nn as nn
import torch.nn.functional as F
from helper_functions import *

class EncoderRNN(nn.Module):
    def __init__(self, cell_type, input_size, embedding_size, hidden_size, num_layers, bidirectional, dropout, device):
        super(EncoderRNN, self).__init__()
        CELL = get_cell_type(cell_type)
        self.cell_type = cell_type
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

    def forward(self, input, hidden, cell_state):
        embedded = self.embedding(input).view(1, 1, -1)
        if self.cell_type != 'LSTM':
            output, hidden = self.cell(embedded, hidden)
            return output, hidden
        else:
            output, (hidden, cell_state) = self.cell(embedded, (hidden, cell_state))
            return output, hidden, cell_state

    def initHidden(self):
        return torch.zeros(self.bidirectional_size*self.num_layers, 1, self.hidden_size, device=self.device)
    
    def initCellState(self):
        return torch.zeros(self.bidirectional_size*self.num_layers, 1, self.hidden_size, device=self.device)
    
class DecoderRNN(nn.Module):
    def __init__(self, cell_type, output_size, embedding_size, hidden_size, num_layers, bidirectional, dropout, device):
        super(DecoderRNN, self).__init__()
        CELL = get_cell_type(cell_type)
        self.cell_type = cell_type
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True if bidirectional == 'True' else False
        self.bidirectional_size = 2 if self.bidirectional else 1
        self.dropout = dropout
        self.device = device
        self.attention = False

        self.embedding = nn.Embedding(output_size, self.embedding_size)
        self.cell = CELL(input_size=embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout, bidirectional=self.bidirectional)
        self.out = nn.Linear(self.bidirectional_size*hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.device = device

    def forward(self, input, hidden, cell_state):
        # IN THE BLOG THEY HAVE ADDED DROPOUT HERE, DO IT?
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        if self.cell_type != 'LSTM':
            output, hidden = self.cell(output, hidden)
            output = self.softmax(self.out(output[0]))
            return output, hidden
        else:
            output, (hidden, cell_state) = self.cell(output, (hidden, cell_state))
            output = self.softmax(self.out(output[0]))
            return output, hidden, cell_state

    def initHidden(self):
        return torch.zeros(self.bidirectional_size*self.num_layers, 1, self.hidden_size, device=self.device)
    
    def initCellState(self):
        return torch.zeros(self.bidirectional_size*self.num_layers, 1, self.hidden_size, device=self.device)
    
class AttnDecoderRNN(nn.Module):
    def __init__(self, cell_type, output_size, embedding_size, hidden_size, num_layers, bidirectional, dropout, device):
        super(AttnDecoderRNN, self).__init__()
        CELL = get_cell_type(cell_type)
        self.cell_type = cell_type
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True if bidirectional == 'True' else False
        self.bidirectional_size = 2 if self.bidirectional else 1
        self.dropout = dropout
        self.device = device
        # self.dropout_p = dropout_p
        # self.max_length = max_length
        self.max_length = 30
        self.attention = True

        self.embedding = nn.Embedding(self.output_size, self.embedding_size)
        self.cell = CELL(input_size=embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout, bidirectional=self.bidirectional)
        self.attn = nn.Linear(self.hidden_size + self.embedding_size, self.max_length)
        self.attn_combine = nn.Linear(self.bidirectional_size*self.hidden_size + self.embedding_size, self.embedding_size)
        # self.dropout = nn.Dropout(self.dropout_p)
        self.out = nn.Linear(self.bidirectional_size*self.hidden_size, self.output_size)  

    def forward(self, input, hidden, encoder_outputs, cell_state):
        embedded = self.embedding(input).view(1, 1, -1)
        # embedded = self.dropout(embedded)
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        if self.cell_type != 'LSTM':
            output, hidden = self.cell(output, hidden)
            output = F.log_softmax(self.out(output[0]), dim=1)
            return output, hidden, attn_weights
        else:
            output, (hidden, cell_state) = self.cell(output, (hidden, cell_state))
            output = F.log_softmax(self.out(output[0]), dim=1)
            return output, hidden, attn_weights, cell_state

    def initHidden(self):
        return torch.zeros(self.bidirectional_size*self.num_layers, 1, self.hidden_size, device=self.device)

    def initCellState(self):
        return torch.zeros(self.bidirectional_size*self.num_layers, 1, self.hidden_size, device=self.device)