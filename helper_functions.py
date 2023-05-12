from io import open
import unicodedata
import string
import re
import random
import time
import os
import pandas as pd

from helper_classes import *

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F #import from torch.nn directly?

def readLangs(data_path, lang1='eng', lang2='kan', reverse=False):
    train_path = os.path.join(data_path, lang2, lang2 + '_train')
    valid_path = os.path.join(data_path, lang2, lang2 + '_valid')
    test_path = os.path.join(data_path, lang2, lang2 + '_test')

    train_df = pd.read_csv(train_path, header=None)
    valid_df = pd.read_csv(valid_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    pairs = [(train_df.iloc[i,0], train_df.iloc[i,1]) for i in range(len(train_df))]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Language(lang2)
        output_lang = Language(lang1)
    else:
        input_lang = Language(lang1)
        output_lang = Language(lang2)

    return input_lang, output_lang, pairs

