import os
import pickle
from helper_functions import *
from train import *

model_path = 'D:\IITM Academic Stuff\Sem 8 Books\CS6910\CS6910-Assignment-3\models\LSTM_False_0.358154296875\\'

with open(model_path+'decoder', 'rb') as file:
    decoder = pickle.load(file)
    print('Decoder Loaded...')
with open(model_path+'encoder', 'rb') as file:
    encoder = pickle.load(file)
    print('Encoder Loaded...')
with open(model_path+'inp_lang', 'rb') as file:
    input_lang = pickle.load(file)
    print('Input Lang Loaded...')
with open(model_path+'out_lang', 'rb') as file:
    output_lang = pickle.load(file)
    print('Output Lang Loaded...')
with open(model_path+'test_pairs', 'rb') as file:
    test_pairs = pickle.load(file)
with open(model_path+'config_loss', 'rb') as file:
    config_loss = pickle.load(file)
with open(model_path+'config_max_length', 'rb') as file:
    config_max_length = pickle.load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loss, test_acc, test_attentions = validIters(encoder, decoder, input_lang, output_lang, test_pairs, config_loss, config_max_length, device)
print(test_loss, test_acc)
if test_attentions is not None:
        plt.matshow(test_attentions)
        plt.show()