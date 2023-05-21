import wandb
from train import *
from sweep_configs import get_config
import pickle 
import os

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

  name = 'cell:{}_lr:{}_es:{}_hs:{}_att:{}'.format(config['CELL'], config['LR'], config['EMBEDDING_SIZE'], config['HIDDEN_SIZE'], config['ATTENTION'])
  run.name = name

  full_model, logs, _ = main(args.data_path, args.input_lang, args.output_lang, config, eval_test=False)

  for i in range(len(logs['iters'])):
    wandb.log({
        'epochs': logs['iters'][i],
        'train_acc': logs['train_acc'][i],
        'train_loss': logs['train_loss'][i], 
        'val_acc': logs['val_acc'][i], 
        'val_loss': logs['val_loss'][i]
    })
  

  save_location = args.save_location
  filename = save_location + '{}_{}_{}/'.format(config['CELL'], config['ATTENTION'], logs['val_acc'][-1])
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
  
  wandb.finish()

wandb.agent(sweep_id, function=wandb_train, count=20)