import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import build_model
from utils.tokenizer import build_tokenizer

from trainer import Trainer, TrainingArgs


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


  
  for bs in train_loader:
    print(bs)
    break

  model = RNN(len(all_letters), 32, len(all_categories))
  # model = LSTM(len(all_letters), 128, len(all_categories))
  # model = UniDeepRNN(len(all_letters), 128, len(all_categories), num_layers=4)
  # model = BiDeepRNN(len(all_letters), 128, len(all_categories), num_layers=4)
  model.to(device)
  trainer = Trainer(
    model=model, 
    training_args=training_args, 
    train_loader=train_loader,
    val_loader=val_loader
  )

  trainer.train()

if __name__ == "__main__":
  main()