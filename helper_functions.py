import random
import time
import os
import pandas as pd
import torch
from torch.optim import Adam, Adadelta, NAdam, RMSprop, SGD, Adagrad
import torch.nn as nn
from torch.nn import LSTM, GRU, RNN
from torch.nn import NLLLoss, CrossEntropyLoss

import time
import math

SOS_token = 0
EOS_token = 1

class Language:
    def __init__(self, name):
        self.name = name
        self.char2index = {}
        self.char2count = {}
        self.index2char = {0: "SOS", 1: "EOS"}
        self.n_chars = 2

    def addWord(self, word):
        for char in word:
            self.addChar(char)

    def addChar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            self.n_chars += 1
        else:
            self.char2count[char] += 1

def readLangs(data_path, lang1='eng', lang2='kan'):
    train_path = os.path.join(data_path, lang2, lang2 + '_train.csv')
    valid_path = os.path.join(data_path, lang2, lang2 + '_valid.csv')
    test_path = os.path.join(data_path, lang2, lang2 + '_test.csv')

    train_df = pd.read_csv(train_path, header=None)
    valid_df = pd.read_csv(valid_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    pairs = [(train_df.iloc[i,0], train_df.iloc[i,1]) for i in range(len(train_df))]
    input_lang = Language(lang1)
    output_lang = Language(lang2)
    return input_lang, output_lang, pairs

def filterPairs(pairs, MAX_LENGTH):
    return [p for p in pairs if (len(p[0]) <= MAX_LENGTH and len(p[1]) <= MAX_LENGTH)]

def prepareData(data_path, lang1, lang2, MAX_LENGTH):
    input_lang, output_lang, pairs = readLangs(data_path, lang1, lang2)
    print("Read %s word pairs" % len(pairs))
    pairs = filterPairs(pairs, MAX_LENGTH)
    print("Counting chars...")
    for pair in pairs:
        input_lang.addWord(pair[0])
        output_lang.addWord(pair[1])
    print("Counted chars:")
    print(input_lang.name, input_lang.n_chars)
    print(output_lang.name, output_lang.n_chars)
    return input_lang, output_lang, pairs

def indexesFromChar(lang, word):
    return [lang.char2index[char] for char in word]

def tensorFromChar(lang, word, device):
    indexes = indexesFromChar(lang, word)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(input_lang, output_lang, pair, device):
    input_tensor = tensorFromChar(input_lang, pair[0], device)
    target_tensor = tensorFromChar(output_lang, pair[1], device)
    return (input_tensor, target_tensor)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio, max_length, device):
    encoder_hidden = encoder.initHidden()
    encoder_cell_state = encoder.initCellState()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # ADD BATCH PROCESSING HERE AFTER FINISHING OTHER PARTS

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.bidirectional_size*encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        # print(ei, input_length, input_tensor[ei])
        if encoder.cell_type != 'LSTM':
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden, None)
        else:
            encoder_output, encoder_hidden, encoder_cell_state = encoder(input_tensor[ei], encoder_hidden, encoder_cell_state)

        # print(input_tensor.data, input_length, ei, encoder_output.shape)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    if decoder.cell_type != 'LSTM':
        decoder_hidden = encoder_hidden
    else:
        decoder_hidden = encoder_hidden
        decoder_cell_state = encoder_cell_state

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            # decoder_output, decoder_hidden, decoder_attention = decoder(
            #     decoder_input, decoder_hidden, encoder_outputs)
            if decoder.cell_type != 'LSTM':
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, None)
            else:
                decoder_output, decoder_hidden, decoder_cell_state = decoder(decoder_input, decoder_hidden, decoder_cell_state)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            # decoder_output, decoder_hidden, decoder_attention = decoder(
            #     decoder_input, decoder_hidden, encoder_outputs)
            if decoder.cell_type != 'LSTM':
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, None)
            else:
                decoder_output, decoder_hidden, decoder_cell_state = decoder(decoder_input, decoder_hidden, decoder_cell_state)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    # INSTEAD OF DOING LOSS BACKWARD HERE FOR EACH AND EVERY PAIR, GET A BUNCH OF PAIRS, CALC TOTAL LOSS AND DO LOSS BACKWARD. 
    # DOING THIS WILL ENABLE BETTER OPTIMIZERS LIKE ADAM TO BE VIABLE

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def get_optimizer(opt):
    if opt == 'Adam':
        return Adam
    elif opt == 'Adadelta':
        return Adadelta
    elif opt == 'Adagrad':
        return Adagrad
    elif opt == 'NAdam':
        return NAdam
    elif opt == 'RMSprop':
        return RMSprop
    elif opt == 'SGD':
        return SGD
    else:
        raise Exception('Incorrect Optimizer')
    
def get_loss_func(loss_func):
    if loss_func == 'CrossEntropyLoss':
        return CrossEntropyLoss
    elif loss_func == 'NLLLoss':
        return NLLLoss
    else:
        raise Exception('Incorrect Loss Function')
    
def get_cell_type(cell):
    if cell == 'LSTM':
        return LSTM
    elif cell == 'RNN':
        return RNN
    elif cell == 'GRU':
        return GRU
    else:
        raise Exception('Incorrect Cell type')


def trainIters(encoder, decoder, input_lang, output_lang, pairs, config, device, print_every=1000):
    LR = config['LR']
    N_ITERS = config['N_ITERS']
    OPT = get_optimizer(config['OPTIM'])
    LOSS_FUNC = get_loss_func(config['LOSS'])
    TFR = config['TFR']
    MAX_LENGTH = config['MAX_LENGTH']

    start = time.time()
    print_loss_total = 0
    print_acc_total = 0

    # ADD BATCH HERE AND IN TRAIN FN IN THE END AFTER FINISHING REST

    encoder_optimizer = OPT(encoder.parameters(), lr=LR)
    decoder_optimizer = OPT(decoder.parameters(), lr=LR)
    training_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(pairs), device) for i in range(N_ITERS)]
    criterion = LOSS_FUNC()

    for iter in range(1, N_ITERS + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, TFR, MAX_LENGTH, device)
        print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / N_ITERS), iter, iter / N_ITERS * 100, print_loss_avg))


def evaluate(encoder, decoder, input_lang, output_lang, word, max_length, device):
    # CHANGE DECODED WORDS TO DECODER CHARS AND SO ON
    with torch.no_grad():
        input_tensor = tensorFromChar(input_lang, word, device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        encoder_cell_state = encoder.initCellState()

        encoder_outputs = torch.zeros(max_length, encoder.bidirectional_size*encoder.hidden_size, device=device)

        for ei in range(input_length):
            if encoder.cell_type != 'LSTM':
                encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden, None)
            else:
                encoder_output, encoder_hidden, encoder_cell_state = encoder(input_tensor[ei], encoder_hidden, encoder_cell_state)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden
        decoder_cell_state = encoder_cell_state

        decoded_words = []
        # decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            # decoder_output, decoder_hidden, decoder_attention = decoder(
            #     decoder_input, decoder_hidden, encoder_outputs)
            # decoder_attentions[di] = decoder_attention.data
            if decoder.cell_type != 'LSTM':
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, None)
            else:
                decoder_output, decoder_hidden, decoder_cell_state = decoder(decoder_input, decoder_hidden, decoder_cell_state)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2char[topi.item()])

            decoder_input = topi.squeeze().detach()

        # return decoded_words, decoder_attentions[:di + 1]
        return decoded_words
    
def evaluateRandomly(encoder, decoder, input_lang, output_lang, pairs, max_length, device, n=20):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        # output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_chars = evaluate(encoder, decoder, input_lang, output_lang, pair[0], max_length, device)
        output_word = ''.join(output_chars)
        print('<', output_word)
        print('')