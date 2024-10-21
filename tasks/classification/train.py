import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import build_model
from utils.tokenizer import build_tokenizer
from trainer import Trainer, TrainingArgs
from dataloader import get_dataloaders

def main():
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
  optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)
  train_loader, val_loader, test_loader = get_dataloaders(
    tokenizer=tokenizer, 
    dataset_args=config["data_config"], 
    training_args=training_args
  )
  
  trainer = Trainer(
    model=model, 
    training_args=training_args, 
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    metric_names=config["metric_config"]["metrics"]
  )

  trainer.train()

if __name__ == "__main__":
  main()