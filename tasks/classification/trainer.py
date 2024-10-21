import tqdm
import numpy as np

import torch
import torch.utils
import torch.nn as nn

from classification.metrics import __all__ as metrics

SUPPORTED_TASKS = ["classification", "causal"]

class TrainingArgs:
  def __init__(
      self, 
      task: str,
      learning_rate: float, 
      training_steps: int,
      metric_log_interval: int,
      training_batch_size: int,
      validation_batch_size: int,
    ):
    """ Training Arguments for the Trainer class

    Args:
        task (str): name of the task
        learning_rate (float): learning rate for the optimizer
        training_steps (int): number of training steps
        metric_log_interval (int): how many steps to wait before logging metrics
        training_batch_size (int): training batch size
        validation_batch_size (int): validation batch size
    """
    assert task in SUPPORTED_TASKS, f"task should be one of {SUPPORTED_TASKS}"
    assert metric_log_interval <= training_steps, "metric_log_interval should be less than or equal to training"
    self.task = task
    self.learning_rate = learning_rate
    self.training_steps = training_steps
    self.metric_log_interval = metric_log_interval
    self.training_batch_size = training_batch_size
    self.validation_batch_size = validation_batch_size

class Trainer:
  def __init__(
      self, 
      model: nn.Module, 
      training_args: TrainingArgs, 
      train_loader: torch.utils.data.DataLoader,
      val_loader: torch.utils.data.DataLoader,
      optimizer: torch.optim.Optimizer,
      metric_names: list[str],
      loss_fn
    ):
    self.args = training_args
    self.model = model
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.metric_names = metric_names
  
  def get_metrics_dict(self):
    return {metric_name: metrics.build(metric_name) for metric_name in self.metric_names}

  def eval_step(self, input, label):
    with torch.no_grad():
      output = self.model(input)
    # outputs : (batch_size, seq_len, num_classes)
    # result : (batch_size, num_classes)
    loss = self.loss_fn(output, label)
    return output, loss.item()

  def eval(self):
    val_loss = []
    eval_metrics_dict = self.get_metrics_dict()
    for input, label in self.val_loader:
      output, loss = self.eval_step(input, label)
      val_loss.append(loss.item()/input.size()[0])
      eval_metrics_dict = {
        metric_name: metric.update(output, label) for metric_name, metric in eval_metrics_dict.items()
      }
    eval_metrics = {
      metric_name: metric.value() for metric_name, metric in eval_metrics.items()
    }
    print(
      f"Validating result:
        Validation Loss: {val_loss / len(val_loss)},
        Metrics: {eval_metrics}"
    )
  
  def train_step(self, input, label):
    self.optimizer.zero_grad()
    output = self.model(input)
    loss = self.loss_fn(output, label)
    loss.backward()
    self.optimizer.step()
    return output, loss.item()

  def train(self):
    self.model.train()
    data_metrics_dict = self.get_metrics_dict()
    for step_id in tqdm.tqdm(range(self.args.training_steps)):
      try:
        input, label = next(data_iter)
      except StopIteration:
        data_iter = iter(self.train_loader)
        input, label = next(data_iter)
      
      output, loss = self.train_step(input, label)
      train_loss += loss
      
      if (step_id + 1) % self.args.metric_log_interval == 0:
        data_metrics_dict = {
          metric_name: metric.update(output, label) for metric_name, metric in data_metrics_dict.items()
        }
        result_metrics = {
          metric_name: metric.value() for metric_name, metric in data_metrics_dict.items()
        }
        print(
          f"Step {step_id + 1}:
            Train Loss: {train_loss / ((step_id + 1) * self.args.training_batch_size) },\n 
            Metrics: {result_metrics}"
        )
      
      if (step_id + 1) % self.args.eval_interval == 0:
        self.eval()