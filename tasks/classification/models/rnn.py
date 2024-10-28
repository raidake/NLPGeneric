import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class RNNLayer(nn.Module):
  def __init__(self, dim_input, dim_hidden, dim_output, direction=1):
    super(RNNLayer, self).__init__()
    self.dim_input = dim_input
    self.dim_hidden = dim_hidden
    
    self.i2h = nn.Linear(dim_input + dim_hidden, dim_hidden)
    self.i2o = nn.Linear(dim_input + dim_hidden, dim_output)
    self.direction = direction

  def forward(self, input, hidden):
    outputs = []
    if self.direction == 1:
      for i in range(input.size()[1]):
        combined = torch.cat((input[:, i, :], hidden), dim=1)
        hidden = self.i2h(combined)
        output_cell = self.i2o(combined)
        outputs.append(output_cell)
    else: 
      for i in range(input.size()[1]-1, -1, -1):
        combined = torch.cat((input[:, i, :], hidden), dim=1)
        hidden = self.i2h(combined)
        output_cell = self.i2o(combined)
        outputs.append(output_cell)
    return torch.stack(outputs, dim=1) # (batch_size, seq_len, dim_output)

  def init_hidden(self, batch_size):
    return torch.zeros(batch_size, self.dim_hidden)

class RNN(nn.Module):
  def __init__(self, vocab_size, dim_input, dim_hidden, dim_output,pretrained_embeddings=None, freeze_embeddings=True):
    super(RNN, self).__init__()

    if pretrained_embeddings is not None:
      print("Loading pretrained word embeddings")
      embedded_matrix = np.load("utils/tokenizer/cache/" + pretrained_embeddings)
      self.token_embedding = nn.Embedding.from_pretrained(
          torch.tensor(embedded_matrix, dtype=torch.float),
          freeze=freeze_embeddings
      )
    else:
      self.token_embedding = nn.Embedding(vocab_size, dim_input)

    self.dim_input = dim_input
    self.dim_hidden = dim_hidden
    self.dim_output = dim_output

    self.rnn_layer = RNNLayer(dim_input, dim_hidden, dim_output)
    self.softmax = nn.LogSoftmax(dim=-1)
  
  def initialize(self):
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)
  
  def forward(self, input):
    hidden = self.rnn_layer.init_hidden(input.size()[0])
    embedded = self.token_embedding(input)
    outputs = self.rnn_layer(embedded, hidden)
    outputs = self.softmax(outputs)
    return outputs

  def init_hidden(self, batch_size):
    return Variable(torch.zeros((batch_size, self.dim_hidden)))