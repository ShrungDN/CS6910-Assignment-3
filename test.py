import os
import pickle
from helper_functions import *
from train import *
import pandas as pd
import numpy as np
import wandb

# This code is used to read a model from a pickle file and then generate predictions on the test data

args = parse_arguments()

model_path = args.load_location
data_path = args.data_path
inp_lang_name = args.input_lang
out_lang_name = args.output_lang
config_max_length = args.max_length

_, _, train_pairs, valid_pairs, test_pairs = prepareData(data_path, inp_lang_name, out_lang_name, config_max_length)

with open(model_path+'/encoder', 'rb') as file:
    encoder = pickle.load(file)
    print('Encoder Loaded...')
with open(model_path+'/decoder', 'rb') as file:
    decoder = pickle.load(file)
    print('Decoder Loaded...')
with open(model_path+'/inp_lang', 'rb') as file:
    input_lang = pickle.load(file)
    print('Input Lang Loaded...')
with open(model_path+'/out_lang', 'rb') as file:
    output_lang = pickle.load(file)
    print('Output Lang Loaded...')
# with open(model_path+'/test_pairs', 'rb') as file:
#     test_pairs = pickle.load(file)
with open(model_path+'/config_loss', 'rb') as file:
    config_loss = pickle.load(file)
# with open(model_path+'/config_max_length', 'rb') as file:
#     config_max_length = pickle.load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# test_loss, test_acc, _ = validIters(encoder, decoder, input_lang, output_lang, test_pairs, config_loss, config_max_length, device)
# print('Test Loss:', test_loss)
# print('Test Accuracy:', test_acc)

if args.wandb_log == 'True':
    import wandb
    ENTITY = args.wandb_entity
    PROJECT = args.wandb_project
    NAME = args.wandb_name

    wandb.login()
    run = wandb.init(entity=ENTITY, project=PROJECT, name=NAME)

    # wandb.log({
        # 'Test Loss': test_loss,
        # 'Test ACcuracy': test_acc})

# # Generating predictions.csv
# test_predicted_pairs = [(p[0], p[1], ''.join(predict(encoder, decoder, input_lang, output_lang, p[0], 30, device))) for p in test_pairs]
# inputs = [p[0] for p in test_predicted_pairs]
# actual = [p[1] for p in test_predicted_pairs]
# preds = [p[2] for p in test_predicted_pairs]

# df = pd.DataFrame({'Input Word': inputs, 
#                    'Actual Output':actual,
#                     'Predicted Output': preds})

# att_str = 'Att' if decoder.attention else 'Vanilla'
# output_file_name = 'predictions_' + att_str + '_' + str(test_acc) + '.csv'
# df.to_csv(output_file_name)

if decoder.attention:
    sample = random.choice(test_pairs)
    pred, att = get_preds_atts(encoder, decoder, input_lang, output_lang, sample[0], config_max_length, device)
    print('Number of decoder layers: ', decoder.num_layers)
    print('Input: ', sample, 'Predicted: ', pred)
    print('Number of characters: Input: ', len(sample[0]), 'Predicted: ', len(sample[1]))
    print('Shape of att: ', att.shape)
    print('shape of att_new: ', att.shape)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(att)
    fig.show()
    plt.show()
    if args.wandb_log == 'True':
        wandb.log({'Attention Map': fig})