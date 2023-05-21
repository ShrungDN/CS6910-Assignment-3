from helper_functions import *
from models import *
from parse_args import parse_arguments
import matplotlib.pyplot as plt
import pickle 

def main(data_path, inp_lang_name, out_lang_name, config, eval_test=False):
    # Function to train a model based on the input hyper parameters
    CELL = config['CELL']
    EMBEDDING_SIZE = config['EMBEDDING_SIZE']
    NUM_LAYERS = config['NUM_LAYERS']
    HIDDEN_SIZE = config['HIDDEN_SIZE']
    BIDIRECTIONAL = config['BIDIRECTIONAL']
    DROPOUT = config['DROPOUT']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)

    # The data is first read from the given data path
    input_lang, output_lang, train_pairs, valid_pairs, test_pairs = prepareData(data_path, inp_lang_name, out_lang_name, config['MAX_LENGTH'])

    # The encoder and decoder objects are formed
    encoder = EncoderRNN(CELL, input_lang.n_chars, EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS, BIDIRECTIONAL, DROPOUT, device).to(device)
    if config['ATTENTION'] == 'False':
        decoder = DecoderRNN(CELL, output_lang.n_chars, EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS, BIDIRECTIONAL, DROPOUT, device).to(device)
    elif config['ATTENTION'] == 'True':
        decoder = AttnDecoderRNN(CELL, output_lang.n_chars, EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS, BIDIRECTIONAL, DROPOUT, device).to(device)
    
    # The encoder-decoder model is trained as per the hyper parameters by using the train_valIters function
    metrics = train_valIters(encoder, decoder, input_lang, output_lang, train_pairs, valid_pairs, config, device, print_every=config['LF'])

    # Randomly selected data points are used to predict outputs.
    predictRandomly(encoder, decoder, input_lang, output_lang, valid_pairs, config['MAX_LENGTH'], device, 20)

    # The model parameters are stored in a dict for future use.
    full_model = {
        'encoder': encoder,
        'decoder': decoder,
        'inp_lang': input_lang,
        'out_lang': output_lang,
        'test_pairs': test_pairs,
        'config_loss': config['LOSS'],
        'config_max_length': config['MAX_LENGTH']
    }

    if not eval_test:
        return full_model, metrics, None
    else:
        # Metrics on the test set are generated if eval_test is set to true
        test_loss, test_acc, test_attentions = validIters(encoder, decoder, input_lang, output_lang, test_pairs, config['LOSS'], config['MAX_LENGTH'], device)
        test_metrics = {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_attentions': test_attentions
        }
        return full_model, metrics, test_metrics
    
if __name__ == '__main__':
    args = parse_arguments()

    config = {
        'CELL':args.cell,
        'EMBEDDING_SIZE':args.embedding_size,
        'NUM_LAYERS':args.num_layers,
        'HIDDEN_SIZE':args.hidden_size,
        'BIDIRECTIONAL':args.bidirectional,
        'DROPOUT':args.dropout,
        'TFR':args.teacher_forcing,
        'MAX_LENGTH':args.max_length,
        'LR':args.learning_rate,
        'N_ITERS':args.epochs,
        'OPTIM':args.optimizer,
        'LOSS':args.loss,
        'LF':args.log_frequency,
        'ATTENTION':args.attention,
    }

    full_model, metrics, test_metrics = main(args.data_path, args.input_lang, args.output_lang, config, eval_test=True)
    print(metrics, test_metrics)

    save_location = args.save_location
    filename = save_location + '{}_{}_{}/'.format(config['CELL'], config['ATTENTION'], metrics['val_acc'][-1])
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename+'encoder', 'wb') as file:
        pickle.dump(full_model['encoder'], file)
    with open(filename+'decoder', 'wb') as file:
        pickle.dump(full_model['decoder'], file)
    with open(filename+'inp_lang', 'wb') as file:
        pickle.dump(full_model['inp_lang'], file)
    with open(filename+'out_lang', 'wb') as file:
        pickle.dump(full_model['out_lang'], file)
    with open(filename+'test_pairs', 'wb') as file:
        pickle.dump(full_model['test_pairs'], file)
    with open(filename+'config_loss', 'wb') as file:
        pickle.dump(full_model['config_loss'], file)
    with open(filename+'config_max_length', 'wb') as file:
        pickle.dump(full_model['config_max_length'], file)

    if test_metrics['test_attentions'] is not None:
        plt.matshow(test_metrics['test_attentions'])
        plt.show()