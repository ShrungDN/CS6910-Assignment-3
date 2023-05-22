# CS6910-Assignment-3
Shrung D N - ME19B168 

WandB Project Link: https://wandb.ai/me19b168/ME19B168_CS6910_Assignment3?workspace=user-me19b168

WandB Report Link: https://wandb.ai/me19b168/ME19B168_CS6910_Assignment3/reports/ME19B168-CS6910-Assignment-3--Vmlldzo0NDI0NjYx

## Description of files:

**helper_functions.py**: 
Python file with helper functions - such as functions used to train the model, load the dataset, transform the dataset, etc.

**models.py**
Python file with model architecture used for the assignment. It consists the code to create the encoder and decoder modules of the seq2seq network.

**parse_args.py**:
Python file used to parse the arguments sent through the command line. It is used to interact with main.py and wandb_train.py via the command line.

**sweep_configs.py**:
Python file with various sweep configurations for hyperparameter tuning using the WandB API.

**test.py**
Python file with the code to import a model from a pickle file and then generate predictions and metrics on the test set
Link to saved best models: https://drive.google.com/drive/folders/18OZoHz3RVa1NRXjJrVVJoM9etb5Ck-_g?usp=sharing

**train.py**:
Python file that is used to train a single model. The hyperparameters used to train this model can be passed as arguments as shown later in this documentation.

**wandb_train.py**:
Python file that makes use of train.py to iteratively train multiple models using different hyperparameters (as defined by the sweep configuration obtained from sweep_configs.py)

Note: The sript to train a particular model is done through the train.py. The script used to tune hyperparameters (by generating sweeps) is done through the wandb_train.py file.

**ME19B168-CS6910-Assignment-3.pdf**
PDF of WandB report for the assignment 

**predictions_attention.csv**
Predictions obtained on the test data using an encoder decoder model with attention.

**predictions_vanilla.csv**
Predictions obtained on the test data using an encoder decoder model without attention.

## train.py Usage
```
usage: python3 train.py [-h --help]
                        [-dp --data_path] <string> Path to directory with training, testing and validation data 
                        [-il --input_lang] <string> Input language name
                        [-ol --output_lang] <string> Output language to be chosen from the different language available
                        [-sl --save_location] <string> File location for saving models 
                        [-es --embedding_size] <int> Size of embedding used for encoder and decoder
                        [-nl --num_layers] <int> Number of recurrent layers in encoder and decoder
                        [-hs --hidden_size] <int> Size of hidden layer in encoder and decoder
                        [-c --cell] <"LSTM", "GRU", "RNN"> Type of cell to be used 
                        [-bi --bidirectional] <"True", "False"> The cell is bidirectional if set to True
                        [-dr --dropout] <float> Dropout parameter to be used during training
                        [-tfr --teacher_forcing] <float> Teacher forcing probability to be used for training
                        [-ml --max_length] <int> Max length of words used for training
                        [-lr --learning_rate] <float> Learning rate used for training
                        [-e --epochs] <int> Number of updates to the parameters during the training process 
                        [-opt --optimizer] <"Adam", "Adadelta", "Adagrad", "NAdam", "RMSprop", "SGD"> Optimizer used for training
                        [-l --loss] <"CrossEntropyLoss", "NLLLoss"> Loss criterion used for training
                        [-lf --log_frequency] <int> Frequency of logging metrics onto wandb
                        [-att --attention] <"True", "False"> Attention decoder is used if set to True	
```

## test.py Usage
```
usage: python3 test.py [-h --help]
                       [-wl --wandb_log] <"True", "False> Logs results on wandb if True
                       [-wp --wandb_project] <string> Name of WandB Projec
                       [-we --wandb_entity] <string> Username of WandB user
                       [-wn --wandb_name] <string> Name of WandB run
                       [-dp --data_path] <string> Path to directory with training, testing and validation data 
                       [-ll --load_location] <string> File location used for loading models 
                       [-il --input_lang] <string> Input language name
                       [-ol --output_lang] <string> Output language to be chosen from the different language available
                       [-ml --max_length] <int> Max length of words used for training
```

## wandb_train.py Usage
```
usage: python3 wandb_train.py [-h --help]
                              [-dp --data_path] <string> Path to directory with training, testing and validation data 
                              [-il --input_lang] <string> Input language name
                              [-ol --output_lang] <string> Output language to be chosen from the different language available
                              [-sl --save_location] <string> File location for saving models 
                              [-wp --wandb_project] <string> Name of WandB Projec
                              [-we --wandb_entity] <string> Username of WandB user
                              [-wn --wandb_name] <string> Name of WandB run
                              [-lf --log_frequency] <int> Frequency of logging metrics onto wandb
                              [-sc --sweep_config] <"SC1", "SC2", "SC3", "SC4", "SC5", "SC6"> Sweep config to be used with wandb_train.py
```

## Method used for generating Predictions:
1. Best hyperparameters are found from sweeping using wandb_train.py - during sweeping, the models are saved
2. More models are trained using train.py using the best hyper parameters (as weight initialization is very important in determining model accuracy)
3. These models are saved
4. The saved models are loaded using test.py, which generates metrics and predictions on the test data

## Optimal Hyperparameters - Without Attention:
```
Cell Type: LSTM
Embedding Layer Size: 128
Hidden Layer Size: 256
Number of Encoder-Decoder Layers: 2
Bidirectional: True
Dropout: 0
Learning Rate: 0.01
Epochs/Iters: 60000
```

## Optimal Hyperparameters - With Attention:
```
Cell Type: LSTM
Embedding Layer Size: 256
Hidden Layer Size: 256
Number of Encoder-Decoder Layers: 2
Bidirectional: True
Dropout: 0.2
Learning Rate: 0.01
Epochs/Iters: 50000
```