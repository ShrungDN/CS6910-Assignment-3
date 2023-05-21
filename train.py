from helper_functions import *
from models import *
from parse_args import parse_arguments
import matplotlib.pyplot as plt

def main(data_path, inp_lang_name, out_lang_name, config, eval_test=False):
    CELL = config['CELL']
    EMBEDDING_SIZE = config['EMBEDDING_SIZE']
    NUM_LAYERS = config['NUM_LAYERS']
    HIDDEN_SIZE = config['HIDDEN_SIZE']
    BIDIRECTIONAL = config['BIDIRECTIONAL']
    DROPOUT = config['DROPOUT']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)

    input_lang, output_lang, train_pairs, valid_pairs, test_pairs = prepareData(data_path, inp_lang_name, out_lang_name, config['MAX_LENGTH'])

    encoder = EncoderRNN(CELL, input_lang.n_chars, EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS, BIDIRECTIONAL, DROPOUT, device).to(device)
    if config['ATTENTION'] == 'False':
        decoder = DecoderRNN(CELL, output_lang.n_chars, EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS, BIDIRECTIONAL, DROPOUT, device).to(device)
    elif config['ATTENTION'] == 'True':
        decoder = AttnDecoderRNN(CELL, output_lang.n_chars, EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS, BIDIRECTIONAL, DROPOUT, device).to(device)

    metrics = train_valIters(encoder, decoder, input_lang, output_lang, train_pairs, valid_pairs, config, device, print_every=config['LF'])

    # predictRandomly(encoder, decoder, input_lang, output_lang, valid_pairs, config['MAX_LENGTH'], device, 20)

    if not eval_test:
        return metrics, None
    else:
        test_loss, test_acc, test_attentions = validIters(encoder, decoder, input_lang, output_lang, test_pairs, config['LOSS'], config['MAX_LENGTH'], device)
        test_metrics = {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_attentions': test_attentions
        }
        return metrics, test_metrics
    
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

    metrics, test_metrics = main(args.data_path, args.input_lang, args.output_lang, config, eval_test=True)
    print(metrics, test_metrics)
    if test_metrics is not None:
        plt.matshow(test_metrics['test_attentions'])
        plt.show()