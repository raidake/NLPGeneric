import json
import argparse
from functools import partial
from typing import Dict

from datasets import load_dataset

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from utils.tokenizer import build_tokenizer, BaseTokenizer
from models import build_model
from trainer import TrainingArgs

SUPPORTED_TASKS = ["classification"]

def pad_collate(batch),:
  (xx, yy) = zip(*batch)
  xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
  return xx_pad, torch.tensor(yy)
class ClassificationDataset(torch.utils.data.Dataset):
  def __init__(self, dataset, tokenizer):
    self.dataset = dataset
    self.tokenizer = tokenizer
  
  def __len__(self):
    return len(self.dataset)
  
  def __getitem__(self, idx):
    item = self.dataset[idx]
    text = item["text"]
    label = item["label"]
    return self.tokenizer(text), label

def get_dataloaders(
  tokenizer: BaseTokenizer,
  dataset_args: Dict,
  training_args: Dict
):
  assert dataset_args["task"] in SUPPORTED_TASKS, f"Task {dataset_args['task']} not supported"
  assert "batch_size" in training_args, "Batch size not found in training args"
  assert "is_huggingface" in dataset_args, "is_huggingface not found in dataset args"
  assert "name" in dataset_args, "Dataset name not found in dataset args"
  
  bs = training_args["batch_size"]
  if dataset_args["is_huggingface"]:
    dataset = load_dataset(dataset_args["name"])
  else:
    assert "path" in dataset_args, "Path not found in dataset args"
    dataset = load_dataset(dataset_args["path"])
  
  if dataset_args["task"] == "classification":
    train_dataset = ClassificationDataset(dataset["train"], tokenizer)
    validation_dataset = ClassificationDataset(dataset["validation"], tokenizer)
    test_dataset = ClassificationDataset(dataset["test"], tokenizer)
    # partial function to be used in DataLoader
    def padding_fn(tokenizer, batch):
      (xx, yy) = zip(*batch)
      xx_pad = pad_sequence(xx, batch_first=True, padding_value=tokenizer.pad_id)
      return xx_pad, torch
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, collate_fn=partial(padding_fn, tokenizer=tokenizer))
    val_loader   = DataLoader(validation_dataset, batch_size=bs, shuffle=True, collate_fn=pad_collate)
    test_loader  = DataLoader(test_dataset, batch_size=bs, shuffle=True, collate_fn=pad_collate)
  else:
    raise NotImplementedError(f"Task {dataset_args['task']} not implemented")
  
  return train_loader, val_loader, test_loader

if __name__ == "__main__":
  argparser = argparse.ArgumentParser()
  argparser.add_argument("--config", type=str, required=True)
  args = argparser.parse_args()
  print("Config file: ", args.config)
  config = json.load(open(args.config))
  print(config)
  
  training_args = TrainingArgs(
    **config["trainer_args"]
  )
  model = build_model(config["model_config"])
  tokenizer = build_tokenizer(config["tokenizer_config"])
  train_loader, val_loader, test_loader = get_dataloaders(
    tokenizer=tokenizer, 
    dataset_args=config["data_config"], 
    training_args=training_args
  )