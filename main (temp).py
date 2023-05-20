from helper_functions import *
from model import CNNModel
from parse_args import parse_arguments

from torchsummary import summary

def main(config, train_data_path, test_data_path, evaluate_model=False):
  IMGDIMS = config['IMGDIMS']
  MEAN, STD = config['MEAN_STD']
  DATA_AUG = config['DATA_AUG']
  BATCH_SIZE = config['BATCH_SIZE']
  LR = config['LR']
  EPOCHS = config['EPOCHS']
  OPTIM = get_optimizer(config['OPTIM'])
  LOSS_FUNC = get_loss_func(config['LOSS_FUNC'])

  model_config = {'IMGDIMS': config['IMGDIMS'],
                  'ACTIVATION': get_activation(config['ACTIVATION']),
                  'POOL': get_pooling(config['POOL']),
                  'NFC': config['NFC'],
                  'BN': config['BN'],
                  'DROPOUT': config['DROPOUT'],
                  'NUM_FILTERS': config['NUM_FILTERS'],
                  'SIZE_FILTERS': config['SIZE_FILTERS']}

  device = ('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"Device: {device}\n")

  train_transform, val_test_transform = get_transforms(DATA_AUG, IMGDIMS, MEAN, STD)
  train_loader, val_loader, test_loader, class_to_idx = get_data_loaders(train_data_path, train_transform, test_data_path, val_test_transform, BATCH_SIZE)

  model = CNNModel(model_config)
  model.to(device, non_blocking=True)
  summary(model, (3, IMGDIMS[0], IMGDIMS[1]))
  print()

  optimizer = OPTIM(model.parameters(), lr=LR)
  criterion = LOSS_FUNC()

  logs = {
     'epochs': [],
     'train_loss': [],
     'train_acc': [],
     'val_loss': [],
     'val_acc': []
  }

  for epoch in range(EPOCHS):
    print(f"Training: Epoch {epoch+1} / {EPOCHS}")

    train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, criterion, device)
    print(f'Training: Loss = {train_epoch_loss:.4f} Accuracy = {train_epoch_acc:.4f}')

    val_epoch_loss, val_epoch_acc = validate(model, val_loader, criterion, device)
    print(f'Validation: Loss = {val_epoch_loss:.4f} Accuracy = {val_epoch_acc:.4f}')

    logs['epochs'].append(epoch + 1)
    logs['train_loss'].append(train_epoch_loss)
    logs['train_acc'].append(train_epoch_acc)
    logs['val_loss'].append(val_epoch_loss)
    logs['val_acc'].append(val_epoch_acc)
    print('-'*50)

  if evaluate_model:
    model_metrics = eval_model(model, train_loader, val_loader, test_loader, criterion, device)
  else:
    model_metrics = [None, None, None]
  return model, logs, model_metrics, class_to_idx, test_loader

if __name__ == '__main__':
  args = parse_arguments()

  config = {'IMGDIMS': (args.dimsw, args.dimsh),
            'BATCH_SIZE': args.batch_size,
            'MEAN_STD': ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            'DATA_AUG': args.data_aug,
            'LR': args.learning_rate,
            'EPOCHS': args.epochs,
            'OPTIM': args.optimizer,
            'LOSS_FUNC': args.loss,
            'DROPOUT': args.dropout,
            'ACTIVATION': args.activation,
            'NFC': args.num_fc,
            'POOL': args.pool,
            'BN': args.batch_norm,
            'NUM_FILTERS': [args.num_filters1, args.num_filters2, args.num_filters3, args.num_filters4, args.num_filters5],
            'SIZE_FILTERS': [args.size_filters1, args.size_filters2, args.size_filters3, args.size_filters4, args.size_filters5]
            }
  
  model, logs, model_metrics, class_to_idx, test_loader = main(config, args.train_data_path, args.test_data_path, evaluate_model=True)

  print('Final Model Metrics:')
  print('Training: Accuracy = {} Loss = {}'.format(model_metrics['train_acc'], model_metrics['train_loss']))
  print('Validation: Accuracy = {} Loss = {}'.format(model_metrics['val_acc'], model_metrics['val_loss']))
  print('Testing: Accuracy = {} Loss = {}'.format(model_metrics['test_acc'], model_metrics['test_loss']))
  print()

  if args.wandb_log == 'True':
    import wandb
    ENTITY = args.wandb_entity
    PROJECT = args.wandb_project
    NAME = args.wandb_name

    wandb.login()
    run = wandb.init(entity=ENTITY, project=PROJECT, name=NAME)

    wandb.log({
          'BATCH_SIZE': config['BATCH_SIZE'],
          'DATA_AUG': config['DATA_AUG'],
          'LR': config['LR'],
          'EPOCHS': config['EPOCHS'],
          'OPTIM': config['OPTIM'],
          'LOSS_FUNC': config['LOSS_FUNC'],
          'DROPOUT': config['DROPOUT'],
          'ACTIVATION': config['ACTIVATION'],
          'NFC': config['NFC'],
          'POOL': config['POOL'],
          'BN': config['BN'],
          'NF1': args.num_filters1,
          'NF2': args.num_filters2,
          'NF3': args.num_filters3,
          'NF4': args.num_filters4,
          'NF5': args.num_filters5,
          'SF1': args.size_filters1,
          'SF2': args.size_filters2,
          'SF3': args.size_filters3,
          'SF4': args.size_filters4,
          'SF5': args.size_filters5
    })

    for i in range(len(logs['epochs'])):
      wandb.log({
          'epochs': logs['epochs'][i],
          'train_acc': logs['train_acc'][i],
          'train_loss': logs['train_loss'][i], 
          'val_acc': logs['val_acc'][i], 
          'val_loss': logs['val_loss'][i]
      })

    wandb.log({'Train Accuracy': model_metrics['train_acc']})
    wandb.log({'Validation Accuracy': model_metrics['val_acc']})
    wandb.log({'Test Accuracy': model_metrics['test_acc']})
    
    if args.view_preds == 'True':
      preds_plot = get_preds_plot(model, test_loader, class_to_idx)
      wandb.log({'Predictions': wandb.Image(preds_plot)})
      preds_plot.savefig(f'ME19B168_PartA_{NAME}_preds_plot')
    
    if args.visualize_filters == 'True':
      filters_plot = get_filters_plot(model)
      wandb.log({'Filters': wandb.Image(filters_plot)})
      filters_plot.savefig(f'ME19B168_PartA_{NAME}_filters_plot')

    wandb.finish()    