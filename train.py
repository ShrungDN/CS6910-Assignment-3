from helper_functions import *
from models import *
from parse_args import parse_arguments

def main(data_path, inp_lang_name, out_lang_name, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_lang, output_lang, pairs = prepareData(data_path, inp_lang_name, out_lang_name, config['MAX_LENGTH'])

    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_chars, hidden_size, device).to(device)
    # attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_chars, dropout_p=0.1).to(device)
    decoder1 = DecoderRNN(hidden_size, output_lang.n_chars, device).to(device)

    # trainIters(encoder1, attn_decoder1, 75000, print_every=5000)
    # return loss acc etc here
    trainIters(encoder1, decoder1, input_lang, output_lang, pairs, config, device, print_every=100)

    # return other stuff here
    evaluateRandomly(encoder1, decoder1, input_lang, output_lang, pairs, device)

if __name__ == '__main__':
    args = parse_arguments()

    config = {
        'CELL':args.cell,
        'EMBEDDING':args.embedding_size,
        'NUM_ENCODER':args.num_encoder,
        'NUM_DECODER':args.num_decoder,
        'HIDDEN_SIZE':args.hidden_size,
        'BIDIRECTIONAL':args.bidirectional,
        'DROPOUT':args.dropout,
        'TFR':args.teacher_forcing,
        'MAX_LENGTH':args.max_length,
        'LR':args.learning_rate,
        'N_ITERS':args.epochs,
        'OPTIM':args.optimizer,
        'LOSS':args.loss,
        # '':,
        # '':,
        # '':,
    }

    _ = main(args.data_path, args.input_lang, args.output_lang, config)