import os
import pickle
from helper_functions import *
from train import *
import pandas as pd

# This code is used to read a model from a pickle file and then generate predictions on the test data

args = parse_arguments()

model_path = args.load_location

with open(model_path+'encoder', 'rb') as file:
    encoder = pickle.load(file)
    print('Encoder Loaded...')
with open(model_path+'decoder', 'rb') as file:
    decoder = pickle.load(file)
    print('Decoder Loaded...')
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
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)
# predictRandomly(encoder, decoder, input_lang, output_lang, test_pairs, config_max_length, device, 30)

# if test_attentions is not None:
#     with open(args.load_location + 'attentions', 'wb') as file:
#       pickle.dump(test_attentions.numpy(), file)

test_predicted_pairs = [(p[0], p[1], ''.join(predict(encoder, decoder, input_lang, output_lang, p[0], 30, device))) for p in test_pairs]
inputs = [p[0] for p in test_predicted_pairs]
actual = [p[1] for p in test_predicted_pairs]
preds = [p[2] for p in test_predicted_pairs]

df = pd.DataFrame({'Input Word': inputs, 
                   'Actual Output':actual,
                    'Predicted Output': preds})
df.to_csv('Predictions.csv')