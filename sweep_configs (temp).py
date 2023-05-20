# Various Sweep Configurations done:
# Some sweep configs are divided into subparts so as to run the sweep parallelly

# Sweeps 1-8:
SC1_1 = {
    'name': 'SC1',
    'method': 'grid',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'epochs': {'values': [10]},
        'batch_size': {'values': [64]},
        'optimizer': {'values': ['Adam']},
        'pool': {'values': ['MaxPool2d']},
        'nfc': {'values': [1000]},
        'lr': {'values': [1e-3]},
        'data_aug': {'values': ['False']},
        'dropout': {'values': [0, 0.3]},
        'activation': {'values': ['ReLU']},
        'batch_norm': {'values': ['True', 'False']},
        'nf1': {'values': [64]},
        'nf2': {'values': [64]},
        'nf3': {'values': [64]},
        'nf4': {'values': [64]},
        'nf5': {'values': [64]},
        'sf1': {'values': [5]},
        'sf2': {'values': [5]},
        'sf3': {'values': [5]},
        'sf4': {'values': [5]},
        'sf5': {'values': [5]}
    }
}

SC1_2 = {
    'name': 'SC1',
    'method': 'grid',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'epochs': {'values': [10]},
        'batch_size': {'values': [64]},
        'optimizer': {'values': ['Adam']},
        'pool': {'values': ['MaxPool2d']},
        'nfc': {'values': [1000]},
        'lr': {'values': [1e-4]},
        'data_aug': {'values': ['False']},
        'dropout': {'values': [0, 0.3]},
        'activation': {'values': ['ReLU']},
        'batch_norm': {'values': ['True', 'False']},
        'nf1': {'values': [64]},
        'nf2': {'values': [64]},
        'nf3': {'values': [64]},
        'nf4': {'values': [64]},
        'nf5': {'values': [64]},
        'sf1': {'values': [5]},
        'sf2': {'values': [5]},
        'sf3': {'values': [5]},
        'sf4': {'values': [5]},
        'sf5': {'values': [5]}
    }
}

SC1_3 = {
    'name': 'SC1',
    'method': 'grid',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'epochs': {'values': [10]},
        'batch_size': {'values': [64]},
        'optimizer': {'values': ['Adam']},
        'pool': {'values': ['MaxPool2d']},
        'nfc': {'values': [1000]},
        'lr': {'values': [1e-5]},
        'data_aug': {'values': ['False']},
        'dropout': {'values': [0, 0.3]},
        'activation': {'values': ['ReLU']},
        'batch_norm': {'values': ['True', 'False']},
        'nf1': {'values': [64]},
        'nf2': {'values': [64]},
        'nf3': {'values': [64]},
        'nf4': {'values': [64]},
        'nf5': {'values': [64]},
        'sf1': {'values': [5]},
        'sf2': {'values': [5]},
        'sf3': {'values': [5]},
        'sf4': {'values': [5]},
        'sf5': {'values': [5]}
    }
}