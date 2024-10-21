from datasets import load_dataset

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def pad_collate(batch):
  (xx, yy) = zip(*batch)
  xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
  return xx_pad, torch.tensor(yy)

def build_dataloaders():
  train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=pad_collate)
  val_loader   = DataLoader(validation_dataset, batch_size=8, shuffle=True, collate_fn=pad_collate)
  test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=True, collate_fn=pad_collate)

if __name__ == "__main__":
    dataset  = load_dataset("rotten_tomatoes")
    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]
    test_dataset = dataset["test"]