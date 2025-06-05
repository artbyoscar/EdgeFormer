from dataclasses import dataclass
import torch.nn as nn
import torch

@dataclass
class EdgeFormerOutput:
    """Common output type for EdgeFormer components"""
    last_hidden_state: torch.Tensor
    past_key_value: tuple = None
    hidden_states: tuple = None
    attentions: tuple = None
