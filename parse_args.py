import argparse

def parse_arguments():
    # function used to parse arguments given through the command line
    parser = argparse.ArgumentParser()

    parser.add_argument('-dp', '--data_path', type=str, default='./data/aksharantar_sampled/', help='Path to data used')
    parser.add_argument('-il', '--input_lang', type=str, default='eng', help='Input Language name')
    parser.add_argument('-ol', '--output_lang', type=str, default='kan', help='Output Language name')
    parser.add_argument('-sl', '--save_location', type=str, default='./models/', help='file location for saving models')
    parser.add_argument('-ll', '--load_location', type=str, default='./models/', help='file location for saving models')

    parser.add_argument('-wp', '--wandb_project', type=str, default='ME19B168_CS6910_Assignment3', help='Project name on WandB')
    parser.add_argument('-we', '--wandb_entity', type=str, default='ME19B168', help='Username on WandB')
    parser.add_argument('-wn', '--wandb_name', type=str, default='ME19B168', help='Display name of run on WandB')
    parser.add_argument('-wl', '--wandb_log', type=str, default='False', help='If "True", results are logged into WandB, specified by wandb_project and wandb_entity')
         
    parser.add_argument('-es', '--embedding_size', type=int, default=128, help='Embedding Size')
    parser.add_argument('-nl', '--num_layers', type=int, default=2, help='Number of Encoder-Decoder Layers')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256, help='Hidden Layer Size')
    parser.add_argument('-c', '--cell', type=str, default='LSTM', help='Cell type: "RNN", "GRU" or "LSTM"')
    parser.add_argument('-bi', '--bidirectional', type=str, default='True', help='Biderictional Cell: "True" or "False')
    parser.add_argument('-dr', '--dropout', type=float, default=0.0, help='Dropout parameter between 0 and 1')
    parser.add_argument('-tfr', '--teacher_forcing', type=float, default=0.5, help='Teacher forcing ratio to be used')
    parser.add_argument('-ml', '--max_length', type=int, default=30, help='Max length of training words to be used')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Learning rate used for training')
    parser.add_argument('-e', '--epochs', type=int, default=60000, help='Number of epochs to train for')
    parser.add_argument('-opt', '--optimizer', type=str, default='SGD', help='Optimizer used for training: "Adam", "Adadelta". "Adagrad", "NAdam", "RMSprop" or "SGD"')
    parser.add_argument('-l', '--loss', type=str, default='NLLLoss', help='Loss function used for training: "CrossEntropyLoss", or "NLLLoss"')
    parser.add_argument('-lf', '--log_frequency', type=int, default=10000, help='Number of iters required for next log')
    parser.add_argument('-att', '--attention', type=str, default='False', help='Whether to choose attention mechanism')

    parser.add_argument('-sc', '--sweep_config', type=str, default='SC1', help='Used with wandb_train.py to choose which sweep config to use')
    
    args = parser.parse_args()
    return args