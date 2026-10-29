from constants import dropout_rate
import torch
from torch import nn

class CNNDigitModel(nn.Module):
    def __init__(self, input_size=28 * 28, output=10):
        super().__init__()
        self.features = nn.Sequential(
          nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1), # 28*28*1 ->28*28*8 
          nn.ReLU(),
          nn.MaxPool2d(2,2),# 28*28*2-> 14*14*2
          nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),# 14*14*8 -> 14*14*16
          nn.ReLU(),
          nn.MaxPool2d(2,2) # 14*14*16 -> 7*7*16
        )
        
        self.classifier = nn.Sequential(
          nn.Flatten(),
          nn.Linear(7*7*128,512),
          nn.BatchNorm1d(512),
          nn.ReLU(),
          nn.Dropout(dropout_rate),
          nn.Linear(512,256),
          nn.BatchNorm1d(256),
          nn.ReLU(),
          nn.Dropout(dropout_rate),
          nn.Linear(256,output)
        )
        
        self.loss_function = nn.CrossEntropyLoss()
    def forward(self,x,labels=None):
      out = self.features(x)
      out = self.classifier(out)
      
      if labels is not None:
        loss = self.loss_function(out,labels)
      else:
        loss = None
        
      return out,loss