import wandb
from train import *
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
  
  config = {
        'CELL':wandb.config.cell,
        'EMBEDDING_SIZE':wandb.config.embedding_size,
        'NUM_LAYERS':wandb.config.num_layers,
        'HIDDEN_SIZE':wandb.config.hidden_size,
        'BIDIRECTIONAL':wandb.config.bidirectional,
        'DROPOUT':wandb.config.dropout,
        'TFR':wandb.config.teacher_forcing,
        'MAX_LENGTH':wandb.config.max_length,
        'LR':wandb.config.learning_rate,
        'N_ITERS':wandb.config.epochs,
        'OPTIM':wandb.config.optimizer,
        'LOSS':wandb.config.loss,
        'LF':args.log_frequency,
        'ATTENTION':wandb.config.attention,
    }

  # name = '{}__bs:{}_da:{}_lr:{}_e:{}_dr:{}_act:{}_nfc:{}_bn:{}_nf:{},{},{},{},{}_sf:{},{},{},{},{}'.format(args.sweep_config,
  #   config['BATCH_SIZE'], config['DATA_AUG'], config['LR'], config['EPOCHS'], config['DROPOUT'], config['ACTIVATION'],
  #   config['NFC'], config['BN'], config['NUM_FILTERS'][0], config['NUM_FILTERS'][1], config['NUM_FILTERS'][2],
  #   config['NUM_FILTERS'][3], config['NUM_FILTERS'][4], config['SIZE_FILTERS'][0], 
  #   config['SIZE_FILTERS'][1], config['SIZE_FILTERS'][2], config['SIZE_FILTERS'][3], config['SIZE_FILTERS'][4]
  # )

  # run.name = name

  logs, _ = main(args.data_path, args.input_lang, args.output_lang, config, eval_test=False)

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