from constants import dropout_rate
from torch import nn


class DigitModel(nn.Module):
    def __init__(self, input_size=28 * 28, output=10):
        super().__init__()
        self.seq_layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, output),
        )
        self.loss_function = nn.CrossEntropyLoss()
    def forward(self,x,labels=None):
      out = self.seq_layers(x)
      if labels is not None:
        loss = self.loss_function(out,labels)
      else:
        loss = None
        
      return out,loss