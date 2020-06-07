#!/usr/bin/env python3
import os
import sys
import argparse
import time
import copy

import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt

# plt.ion()

# Arguments
parser = argparse.ArgumentParser()

parser.add_argument(
  '--log-dir', type=str, default=None,
  help='Debug: Path to writable directory for a log file to be created. Default: log to stdout / stderr'
)
parser.add_argument(
  '--log-file-name', type=str, default='training.log',
  help='Debug: Name of the log file, generated when --log-dir is set. Default: training.log'
)

args = parser.parse_args()

# Redirect output streams for logging
if args.log_dir:
  log_file = open(os.path.join(os.path.expanduser(args.log_dir), args.log_file_name), 'w')
  sys.stdout = log_file
  sys.stderr = log_file

transform = transforms.Compose([
  transforms.ToTensor()
])

#data_dir = 'mnist/MNIST/processed'
train_dataset = datasets.MNIST('mnist', transform=transform, train=True, download=True)
test_dataset = datasets.MNIST('mnist', transform=transform, train=False, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def imshow(inp, title=None):
#   inp = inp.numpy().transpose((1, 2, 0))
#   plt.imshow(inp)
#   if title is not None:
#     plt.title(title)
#   plt.pause(5)
  
train_inputs, train_labels = next(iter(train_loader))
test_inputs, test_labels = next(iter(test_loader))

train_out = torchvision.utils.make_grid(train_inputs)
test_out = torchvision.utils.make_grid(test_inputs)

class Modelaso(nn.Module):
  def __init__(self):
    super(Modelaso, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
    self.pool = nn.MaxPool2d((2, 2))
    self.fc1 = nn.Linear(32 * 14 * 14, 120) # AUF PADDING UND DIMENSIONEN ACHTEN!
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool(x)
    # print(x.shape)
    x = x.view(x.size(0), -1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x


model = Modelaso()
# model.load_state_dict(torch.load('weights.pt'))
model = model.to(device)

def train():
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.01)

  best_loss = 0

  for e in range(10):
    running_loss = 0

    for data in train_loader:

      images, labels = data[0].to(device), data[1].to(device)

      # Training pass
      optimizer.zero_grad()
      
      output = model(images)
      loss = criterion(output, labels)
      
      #This is where the model learns by backpropagating
      loss.backward()
      
      #And optimizes its weights here
      optimizer.step()
      
      running_loss += loss.item()

      if e == 0:
        best_loss = running_loss

    else:
      if running_loss < best_loss:
        best_loss = running_loss
        torch.save(model.state_dict(), 'weights.pt')
      print("Epoch {} - Training loss: {}".format(e, running_loss / len(train_loader)))

train()

def eval():
  correct = 0
  with torch.no_grad():
    for i, data in enumerate(test_loader):
      images, labels = data[0].to(device), data[1].to(device)

      output = model(images)
      _, preds = torch.max(output, 1)

      correct += (len(labels) - len(torch.nonzero(preds - labels)))

  print(correct / len(test_loader.dataset))

eval()

if args.log_dir:
  sys.stdout.close()