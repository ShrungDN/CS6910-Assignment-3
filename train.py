from helper_classes import *
from helper_functions import *
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_forcing_ratio = 0.5
MAX_LENGTH = 30
input_lang, output_lang, pairs = prepareData('./data/aksharantar_sampled/', 'eng', 'kan', True)

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_chars, hidden_size, device).to(device)
# attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_chars, dropout_p=0.1).to(device)
decoder1 = DecoderRNN(hidden_size, output_lang.n_chars, device)

# trainIters(encoder1, attn_decoder1, 75000, print_every=5000)
trainIters(encoder1, decoder1, input_lang, output_lang, pairs, 5000, device, print_every=100)

evaluateRandomly(encoder1, decoder1, input_lang, output_lang, pairs, device)