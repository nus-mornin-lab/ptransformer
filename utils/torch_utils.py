import torch

def param_reset(m):
    if hasattr(m, "reset_parameters"):
        m.reset_parameters()
