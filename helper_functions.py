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
    # Defining a language class to keep track of attributes of the language like number of characters, character to index encoding
    def __init__(self, name):
        self.name = name
        self.char2index = {}
        self.char2count = {}
        self.index2char = {0: "SOS", 1: "EOS"}
        self.n_chars = 2

    def addWord(self, word):
        # Function to read a word and the characters to it to vocabulary
        for char in word:
            self.addChar(char)

    def addChar(self, char):
        # Adding character to vocabulary
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            self.n_chars += 1
        else:
            self.char2count[char] += 1

def readLangs(data_path, lang1='eng', lang2='kan'):
    # Read data from files and generating input and output language classes and pairs
    train_path = os.path.join(data_path, lang2, lang2 + '_train.csv')
    valid_path = os.path.join(data_path, lang2, lang2 + '_valid.csv')
    test_path = os.path.join(data_path, lang2, lang2 + '_test.csv')

    train_df = pd.read_csv(train_path, header=None)
    valid_df = pd.read_csv(valid_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    train_pairs = [(train_df.iloc[i,0], train_df.iloc[i,1]) for i in range(len(train_df))]
    valid_pairs = [(valid_df.iloc[i,0], valid_df.iloc[i,1]) for i in range(len(valid_df))]
    test_pairs = [(test_df.iloc[i,0], test_df.iloc[i,1]) for i in range(len(test_df))]

    input_lang = Language(lang1)
    output_lang = Language(lang2)
    return input_lang, output_lang, train_pairs, valid_pairs, test_pairs

def filterPairs(pairs, MAX_LENGTH):
    # Remove very long words from training to remove outliers
    return [p for p in pairs if (len(p[0]) <= MAX_LENGTH and len(p[1]) <= MAX_LENGTH)]

def prepareData(data_path, lang1, lang2, MAX_LENGTH):
    # Function that combines the entire data pre processing process
    input_lang, output_lang, train_pairs, valid_pairs, test_pairs = readLangs(data_path, lang1, lang2)
    print("Read %s word pairs" % len(train_pairs))
    train_pairs = filterPairs(train_pairs, MAX_LENGTH)
    valid_pairs = filterPairs(valid_pairs, MAX_LENGTH)
    test_pairs = filterPairs(test_pairs, MAX_LENGTH)
    print("Counting chars...")
    for pair in train_pairs:
        input_lang.addWord(pair[0])
        output_lang.addWord(pair[1])
    print("Counted chars:")
    print(input_lang.name, input_lang.n_chars)
    print(output_lang.name, output_lang.n_chars)
    return input_lang, output_lang, train_pairs, valid_pairs, test_pairs

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
    # Function to get optimizer based on string argument
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
    # Function to get loss function based on string argument
    if loss_func == 'CrossEntropyLoss':
        return CrossEntropyLoss
    elif loss_func == 'NLLLoss':
        return NLLLoss
    else:
        raise Exception('Incorrect Loss Function')
    
def get_cell_type(cell):
    # Function to get cell type based on string argument
    if cell == 'LSTM':
        return LSTM
    elif cell == 'RNN':
        return RNN
    elif cell == 'GRU':
        return GRU
    else:
        raise Exception('Incorrect Cell type')

def train_valIters(encoder, decoder, input_lang, output_lang, train_pairs, valid_pairs, config, device, print_every=1000):
    # Trains the model over train and finds metrics on train and validation sets
    # SGD optimizer is used here
    print('Training...')
    # Hyper parameters:
    LR = config['LR']
    N_ITERS = config['N_ITERS']
    OPT = get_optimizer(config['OPTIM'])
    LOSS_FUNC = get_loss_func(config['LOSS'])
    TFR = config['TFR']
    MAX_LENGTH = config['MAX_LENGTH']

    start = time.time()
    print_loss_total = 0
    print_acc_total = 0

    # Initiating optimizer and loss criterion
    encoder_optimizer = OPT(encoder.parameters(), lr=LR)
    decoder_optimizer = OPT(decoder.parameters(), lr=LR)
    training_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(train_pairs), device) for i in range(N_ITERS)]
    criterion = LOSS_FUNC()

    metrics = {
        'iters':[],
        'train_loss':[],
        'train_acc':[],
        'val_loss':[],
        'val_acc':[]
    }

    for iter in range(1, N_ITERS + 1):
        if iter % 1000 == 0:
            print(iter)

        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        # A single pair is sent to the train function, which outputs the accuracy (0 or 1) and corresponging loss
        loss, acc = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, TFR, MAX_LENGTH, device)
        print_loss_total += loss
        print_acc_total += acc

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_acc_avg = print_acc_total / print_every
            print_loss_total = 0
            print_acc_total = 0
            print('Validating...')
            val_loss, val_acc, _ = validIters(encoder, decoder, input_lang, output_lang, valid_pairs, config['LOSS'], MAX_LENGTH, device)
            print('Logging...')
            print('%s (%d %d%%) Training: Loss = %.4f Accuracy = %.04f Validation: Loss = %.4f Accuracy = %.04f' % (timeSince(start, iter / N_ITERS), iter, iter / N_ITERS * 100, print_loss_avg, print_acc_avg, val_loss, val_acc))
            metrics['iters'].append(iter)
            metrics['train_loss'].append(print_loss_avg)
            metrics['train_acc'].append(print_acc_avg)
            metrics['val_loss'].append(val_loss)
            metrics['val_acc'].append(val_acc)
            print('Training...')
    return metrics

def predictRandomly(encoder, decoder, input_lang, output_lang, pairs, max_length, device, n=20):
    # Function to predict randomly selected strings from a given list of pairs
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_chars = predict(encoder, decoder, input_lang, output_lang, pair[0], max_length, device)
        output_word = ''.join(output_chars)
        print('<', output_word)
        print('')

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio, max_length, device):
    # Function that performs the gradient update
    # initialize encoder and decoder states and set parameters required grad to true:
    encoder.train()
    decoder.train()
    encoder_hidden = encoder.initHidden()
    encoder_cell_state = encoder.initCellState()

    # Zero out the gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.bidirectional_size*encoder.hidden_size, device=device)
    loss = 0

    # Calculate the encoder output for the given input - this code uses multiple if else statements to
    # perform the appropriate calculation based on whether the cell is an LSTM or not and whether the 
    # model requires attention or not
    for ei in range(input_length):
        if encoder.cell_type != 'LSTM':
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden, None)
        else:
            encoder_output, encoder_hidden, encoder_cell_state = encoder(input_tensor[ei], encoder_hidden, encoder_cell_state)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoded_chars = []

    # Exchange values between decoder and encoder
    if decoder.cell_type != 'LSTM':
        decoder_hidden = encoder_hidden
    else:
        decoder_hidden = encoder_hidden
        decoder_cell_state = encoder_cell_state

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Using teacher forcing for training:
        for di in range(target_length):
            if decoder.cell_type != 'LSTM':
                if decoder.attention:
                    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, None)
                else:
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, None)
            else:
                if decoder.attention:
                    decoder_output, decoder_hidden, decoder_attention, decoder_cell_state = decoder(decoder_input, decoder_hidden, encoder_outputs, decoder_cell_state)
                else:
                    decoder_output, decoder_hidden, decoder_cell_state = decoder(decoder_input, decoder_hidden, decoder_cell_state)
            topv, topi = decoder_output.topk(1)
            decoded_char = topi.squeeze().detach()
            decoded_chars.append(decoded_char.item())
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]

    else:
        # Not using teach forcing for training:
        for di in range(target_length):
            if decoder.cell_type != 'LSTM':
                if decoder.attention:
                    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, None)
                else:
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, None)
            else:
                if decoder.attention:
                    decoder_output, decoder_hidden, decoder_attention, decoder_cell_state = decoder(decoder_input, decoder_hidden, encoder_outputs, decoder_cell_state)
                else:
                    decoder_output, decoder_hidden, decoder_cell_state = decoder(decoder_input, decoder_hidden, decoder_cell_state)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            decoded_chars.append(decoder_input.item())
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    # Calculat accuracy: 1 if all chars in the predicted word are same as true output
    target_chars = [c.detach().item() for c in target_tensor]
    acc = float(target_chars == decoded_chars)

    loss.backward()

    # Changing the parameters of the model
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length, acc

def validIters(encoder, decoder, input_lang, output_lang, pairs, criterion, max_length, device):
    # Find loss and accuracy of the validation set
    # Requires grad parameters are first set to False 
    encoder.eval()
    decoder.eval()
    criterion = get_loss_func(criterion)
    criterion = criterion()

    # The loss and accuracy of entire valid set is found as follows
    with torch.no_grad():
        print_loss_total = 0
        print_acc_total = 0

        N_ITERS = len(pairs)
        training_pairs = [tensorsFromPair(input_lang, output_lang, pairs[i], device) for i in range(len(pairs))]

        for iter in range(1, N_ITERS + 1):
            if iter % 1000 == 0:
                print(iter)
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            # Each pair in the valid set is sent to the valid function, which returns the loss and accuracy (0 or 1) for each pair
            if decoder.attention:
                loss, acc, attentions = valid(input_tensor, target_tensor, encoder, decoder, criterion, max_length, device)
            else:
                loss, acc, _ = valid(input_tensor, target_tensor, encoder, decoder, criterion, max_length, device)
            print_loss_total += loss
            print_acc_total += acc

        loss = print_loss_total / N_ITERS
        acc = print_acc_total / N_ITERS
        if decoder.attention:
            return loss, acc, attentions
        else:
            return loss, acc, None

def valid(input_tensor, target_tensor, encoder, decoder, criterion, max_length, device):
    # Function used to calculate the loss and accuracy for a given input and output word
    # The function behaves similarly to the train function except for calculating gradients and using teacher forcing (which is not done in validation time)
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        encoder_hidden = encoder.initHidden()
        encoder_cell_state = encoder.initCellState()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(max_length, encoder.bidirectional_size*encoder.hidden_size, device=device)
        loss = 0

        for ei in range(input_length):
            if encoder.cell_type != 'LSTM':
                encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden, None)
            else:
                encoder_output, encoder_hidden, encoder_cell_state = encoder(input_tensor[ei], encoder_hidden, encoder_cell_state)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoded_chars = []
        decoder_attentions = torch.zeros(max_length, max_length)

        if decoder.cell_type != 'LSTM':
            decoder_hidden = encoder_hidden
        else:
            decoder_hidden = encoder_hidden
            decoder_cell_state = encoder_cell_state

        for di in range(target_length):
            if decoder.cell_type != 'LSTM':
                if decoder.attention:
                    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, None)
                else:
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, None)
            else:
                if decoder.attention:
                    decoder_output, decoder_hidden, decoder_attention, decoder_cell_state = decoder(decoder_input, decoder_hidden, encoder_outputs, decoder_cell_state)
                else:
                    decoder_output, decoder_hidden, decoder_cell_state = decoder(decoder_input, decoder_hidden, decoder_cell_state)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach() 
            decoded_chars.append(decoder_input.item())
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

        target_chars = [c.detach().item() for c in target_tensor]
        acc = float(target_chars == decoded_chars)

        if decoder.attention:
            return loss.item() / target_length, acc, decoder_attentions[:di + 1]
        else:
            return loss.item() / target_length, acc, None

def predict(encoder, decoder, input_lang, output_lang, word, max_length, device):
    # Function to predict the output word based on a input word
    encoder.eval()
    decoder.eval()
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

        decoder_input = torch.tensor([[SOS_token]], device=device) 

        decoder_hidden = encoder_hidden
        decoder_cell_state = encoder_cell_state

        decoded_words = []

        for di in range(max_length):
            if decoder.cell_type != 'LSTM':
                if decoder.attention:
                    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, None)
                else:
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, None)
            else:
                if decoder.attention:
                    decoder_output, decoder_hidden, decoder_attention, decoder_cell_state = decoder(decoder_input, decoder_hidden, encoder_outputs, decoder_cell_state)
                else:
                    decoder_output, decoder_hidden, decoder_cell_state = decoder(decoder_input, decoder_hidden, decoder_cell_state)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2char[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words