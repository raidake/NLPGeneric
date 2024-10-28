from typing import Dict
from torch import nn
from models import (
    BiDeepRNN, 
    LSTM, 
    RNN, 
    UniDeepRNN
)

MODULE_MAP = {
    "RNN": RNN,
    "LSTM": LSTM,
    "UniDeepRNN": UniDeepRNN,
    "BiDeepRNN": BiDeepRNN
}

def build_model(config: Dict)-> nn.Module:
    if "model_type" not in config:
        raise Exception("model_type not found in config")
    model_type = config["model_type"]
    if model_type not in MODULE_MAP:
        raise Exception(f"model_type {model_type} not found in MODULE_MAP")
    if "embedding_matrix" in config["args"]:
        import numpy as np
        embedding_matrix = np.load(config["args"]["embedding_matrix"])
        config["args"]["pretrained_embeddings"] = embedding_matrix  # Pass to model
    return MODULE_MAP[model_type](**config["args"])