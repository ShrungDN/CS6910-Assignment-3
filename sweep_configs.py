# Different Sweep configurations used. It is a combination of bayes and random sweeps, with each successive sweep 
# configuration containing better hyper parameter ranges than the previous
SC1 = {
    'name': 'SC1',
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'cell': {'values': ["RNN", "GRU", "LSTM"]},
        'embedding_size': {'values': [64, 128, 256]},
        'num_layers': {'values': [1, 2, 3]},
        'hidden_size': {'values': [64, 128, 256]},
        'bidirectional': {'values': ['True', 'False']},
        'dropout': {'values': [0, 0.2]},
        'teacher_forcing': {'values': [0.5]},
        'max_length': {'values': [30]},
        'learning_rate': {'values': [0.01, 0.001]},
        'epochs': {'values': [50000]},
        'optimizer': {'values': ['SGD']},
        'loss': {'values': ['NLLLoss']},
        'attention': {'values': ['False']},
    }
}

SC2 = {
    'name': 'SC2',
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'cell': {'values': ["RNN", "GRU", "LSTM"]},
        'embedding_size': {'values': [128, 256, 512]},
        'num_layers': {'values': [1, 2, 3]},
        'hidden_size': {'values': [128, 256, 512]},
        'bidirectional': {'values': ['True', 'False']},
        'dropout': {'values': [0, 0.2]},
        'teacher_forcing': {'values': [0.5]},
        'max_length': {'values': [30]},
        'learning_rate': {'values': [0.01, 0.05]},
        'epochs': {'values': [50000]},
        'optimizer': {'values': ['SGD']},
        'loss': {'values': ['NLLLoss']},
        'attention': {'values': ['False']},
    }
}

SC3 = {
    'name': 'SC3',
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'cell': {'values': ["RNN", "GRU", "LSTM"]},
        'embedding_size': {'values': [128, 256, 512]},
        'num_layers': {'values': [1, 2, 3]},
        'hidden_size': {'values': [128, 256, 512]},
        'bidirectional': {'values': ['True', 'False']},
        'dropout': {'values': [0, 0.2]},
        'teacher_forcing': {'values': [0.5]},
        'max_length': {'values': [30]},
        'learning_rate': {'values': [0.01, 0.05]},
        'epochs': {'values': [50000]},
        'optimizer': {'values': ['SGD']},
        'loss': {'values': ['NLLLoss']},
        'attention': {'values': ['True']},
    }
}

SC4 = {
    'name': 'SC4',
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'val_loss'},
    'parameters': 
    {
        'cell': {'values': ["RNN", "GRU", "LSTM"]},
        'embedding_size': {'values': [128, 256, 512]},
        'num_layers': {'values': [1, 2, 3]},
        'hidden_size': {'values': [128, 256, 512]},
        'bidirectional': {'values': ['True', 'False']},
        'dropout': {'values': [0, 0.2]},
        'teacher_forcing': {'values': [0.5]},
        'max_length': {'values': [30]},
        'learning_rate': {'values': [0.01, 0.005]},
        'epochs': {'values': [50000]},
        'optimizer': {'values': ['SGD']},
        'loss': {'values': ['NLLLoss']},
        'attention': {'values': ['True']},
    },
    'early_terminate': {'type': 'hyperband', 'max_iter':5, 's':5, 'eta':1}
}

SC5 = {
    'name': 'SC5',
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'cell': {'values': ["GRU", "LSTM"]},
        'embedding_size': {'values': [128, 256]},
        'num_layers': {'values': [2, 3]},
        'hidden_size': {'values': [128, 256]},
        'bidirectional': {'values': ['True', 'False']},
        'dropout': {'values': [0, 0.1]},
        'teacher_forcing': {'values': [0.5]},
        'max_length': {'values': [30]},
        'learning_rate': {'values': [0.01, 0.02]},
        'epochs': {'values': [50000, 60000, 70000, 80000]},
        'optimizer': {'values': ['SGD']},
        'loss': {'values': ['NLLLoss']},
        'attention': {'values': ['False']},
    }
}

SC6 = {
    'name': 'SC6',
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'cell': {'values': ["GRU", "LSTM"]},
        'embedding_size': {'values': [128, 256]},
        'num_layers': {'values': [2, 3]},
        'hidden_size': {'values': [128, 256]},
        'bidirectional': {'values': ['True']},
        'dropout': {'values': [0.1, 0.2]},
        'teacher_forcing': {'values': [0.5]},
        'max_length': {'values': [30]},
        'learning_rate': {'values': [0.01, 0.02]},
        'epochs': {'values': [50000, 60000, 70000, 80000]},
        'optimizer': {'values': ['SGD']},
        'loss': {'values': ['NLLLoss']},
        'attention': {'values': ['True']},
    }
}

SC7 = {
    'name': 'SC7',
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'cell': {'values': ["LSTM"]},
        'embedding_size': {'values': [256]},
        'num_layers': {'values': [2]},
        'hidden_size': {'values': [256]},
        'bidirectional': {'values': ['True']},
        'dropout': {'values': [0.0]},
        'teacher_forcing': {'values': [0.5]},
        'max_length': {'values': [30]},
        'learning_rate': {'values': [0.01]},
        'epochs': {'values': [60000]},
        'optimizer': {'values': ['SGD']},
        'loss': {'values': ['NLLLoss']},
        'attention': {'values': ['True']},
    }
}

def get_config(name):
    if name == 'SC1':
        return SC1
    elif name == 'SC2':
        return SC2
    elif name == 'SC3':
        return SC3
    elif name == 'SC4':
        return SC4
    elif name == 'SC5':
        return SC5
    elif name == 'SC6':
        return SC6
    elif name == 'SC7':
        return SC7
    # elif name == 'SC4_2':
    #     return SC4_2
    # elif name == 'SC4_3':
    #     return SC4_3
    # elif name == 'SC4_4':
    #     return SC4_4
    # elif name == 'SC4_5':
    #     return SC4_5
    # elif name == 'SC4_6':
    #     return SC4_6
    # elif name == 'SC4_7':
    #     return SC4_7
    # elif name == 'SC4_8':
    #     return SC4_8