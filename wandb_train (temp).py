import wandb
from main import *
from sweep_configs import get_config

args = parse_arguments()

SC = args.sweep_config
CONFIG = get_config(SC)

ENTITY = args.wandb_entity
PROJECT = args.wandb_project
NAME = args.wandb_name

wandb.login()

sweep_id = wandb.sweep(sweep=CONFIG, project=PROJECT)

def wandb_train():
  
  run = wandb.init(entity=ENTITY, project=PROJECT, name=NAME)

  config = {'IMGDIMS': (args.dimsw, args.dimsh),
            'BATCH_SIZE': wandb.config.batch_size,
            'MEAN_STD': ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            'DATA_AUG': wandb.config.data_aug,
            'LR': wandb.config.lr,
            'EPOCHS': wandb.config.epochs,
            'OPTIM': wandb.config.optimizer,
            'LOSS_FUNC': args.loss,
            'DROPOUT': wandb.config.dropout,
            'ACTIVATION': wandb.config.activation,
            'NFC': wandb.config.nfc,
            'POOL': wandb.config.pool,
            'BN': wandb.config.batch_norm,
            'NUM_FILTERS': [wandb.config.nf1, wandb.config.nf2, wandb.config.nf3, wandb.config.nf4, wandb.config.nf5],
            'SIZE_FILTERS': [wandb.config.sf1, wandb.config.sf2, wandb.config.sf3, wandb.config.sf4, wandb.config.sf5]
            }

  name = '{}__bs:{}_da:{}_lr:{}_e:{}_dr:{}_act:{}_nfc:{}_bn:{}_nf:{},{},{},{},{}_sf:{},{},{},{},{}'.format(args.sweep_config,
    config['BATCH_SIZE'], config['DATA_AUG'], config['LR'], config['EPOCHS'], config['DROPOUT'], config['ACTIVATION'],
    config['NFC'], config['BN'], config['NUM_FILTERS'][0], config['NUM_FILTERS'][1], config['NUM_FILTERS'][2],
    config['NUM_FILTERS'][3], config['NUM_FILTERS'][4], config['SIZE_FILTERS'][0], 
    config['SIZE_FILTERS'][1], config['SIZE_FILTERS'][2], config['SIZE_FILTERS'][3], config['SIZE_FILTERS'][4]
  )

  run.name = name

  model, logs, model_metrics, class_to_idx, test_loader = main(config, args.train_data_path, args.test_data_path)

  for i in range(len(logs['epochs'])):
    wandb.log({
        'epochs': logs['epochs'][i],
        'train_acc': logs['train_acc'][i],
        'train_loss': logs['train_loss'][i], 
        'val_acc': logs['val_acc'][i], 
        'val_loss': logs['val_loss'][i]
    })
  
  wandb.finish()

wandb.agent(sweep_id, function=wandb_train, count=50)