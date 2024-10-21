from typing import Dict
from torch import nn
from classification.models import (
    BiDeepRNN, 
    LSTM, 
    RNN, 
    UniDeepRNN
)

def build_model(config: Dict)-> nn.Module:
    if "model_type" not in config:
        raise Exception("model_type not found in config")
    model_type = config["model_type"]
    if model_type == "RNN":
        return RNN(**config["args"])
    elif model_type == "LSTM":
        return LSTM(**config["args"])
    elif model_type == "UniDeepRNN":
        return UniDeepRNN(**config["args"])
    elif model_type == "BiDeepRNN":
        return BiDeepRNN(**config["args"])
    else:
        raise Exception("model_type is not supported")