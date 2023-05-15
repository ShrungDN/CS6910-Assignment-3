import random
import time
import os
import pandas as pd

import time
import math

from helper_classes import *

###############################################################

###############################################################

# GO THROUGH EACH FUNCTION AND EACH SENTENCE ONCE AND MAKE CHANGES APPROPRIATELY 
# CHANGE THE ENCODER DECODER CODE TO HAVE LSTM RNN GRU PROPERLY
###############################################################

###############################################################


import torch.nn.functional as F #import from torch.nn directly?
MAX_LENGTH = 30
teacher_forcing_ratio = 0.5

def readLangs(data_path, lang1='eng', lang2='kan', reverse=False):
    train_path = os.path.join(data_path, lang2, lang2 + '_train.csv')
    valid_path = os.path.join(data_path, lang2, lang2 + '_valid.csv')
    test_path = os.path.join(data_path, lang2, lang2 + '_test.csv')

    train_df = pd.read_csv(train_path, header=None)
    valid_df = pd.read_csv(valid_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    pairs = [(train_df.iloc[i,0], train_df.iloc[i,1]) for i in range(len(train_df))]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Language(lang2)
        output_lang = Language(lang1)
    else:
        input_lang = Language(lang1)
        output_lang = Language(lang2)

    return input_lang, output_lang, pairs

def filterPairs(pairs):
    return [p for p in pairs if (len(p[0]) <= MAX_LENGTH and len(p[1]) <= MAX_LENGTH)]

def prepareData(data_path, lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(data_path, lang1, lang2, reverse)
    print("Read %s word pairs" % len(pairs))
    pairs = filterPairs(pairs)
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

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device, max_length=30):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        # print(ei, input_length, input_tensor[ei])
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        # print(input_tensor.data, input_length, ei, encoder_output.shape)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            # decoder_output, decoder_hidden, decoder_attention = decoder(
            #     decoder_input, decoder_hidden, encoder_outputs)
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            # decoder_output, decoder_hidden, decoder_attention = decoder(
            #     decoder_input, decoder_hidden, encoder_outputs)
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

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

def trainIters(encoder, decoder, input_lang, output_lang, pairs, n_iters, device, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(pairs), device) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, device)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

def evaluate(encoder, decoder, input_lang, output_lang, word, device, max_length=30):
    with torch.no_grad():
        input_tensor = tensorFromChar(input_lang, word, device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            # decoder_output, decoder_hidden, decoder_attention = decoder(
            #     decoder_input, decoder_hidden, encoder_outputs)
            # decoder_attentions[di] = decoder_attention.data
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2char[topi.item()])

            decoder_input = topi.squeeze().detach()

        # return decoded_words, decoder_attentions[:di + 1]
        return decoded_words
    
def evaluateRandomly(encoder, decoder, input_lang, output_lang, pairs, device, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        # output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_chars = evaluate(encoder, decoder, input_lang, output_lang, pair[0], device)
        output_word = ''.join(output_chars)
        print('<', output_word)
        print('')