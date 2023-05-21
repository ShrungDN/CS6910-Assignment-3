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
        'learning_rate': {'values': [0.1, 0.01]},
        'epochs': {'values': [50000]},
        'optimizer': {'values': ['SGD']},
        'loss': {'values': ['NLLLoss']},
        'attention': {'values': ['False']},
    }
}

def get_config(name):
    if name == 'SC1':
        return SC1
    elif name == 'SC2':
        return SC2
    # elif name == 'SC1_3':
    #     return SC1_3
    # elif name == 'SC2':
    #     return SC2
    # elif name == 'SC3':
    #     return SC3
    # elif name == 'SC4_1':
    #     return SC4_1
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