import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mtp

import torch
from torch.utils import data

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, inputs, labels):
        'Initialization'
        self.labels = labels
        self.inputs = inputs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        #input = self.inputs[index]

        # Load data and get label
        X = self.inputs[index]
        y = self.labels[index]

        return X, y, #X_test, y_test