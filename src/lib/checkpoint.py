import torch

def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_checkpoint(filename):
    state = torch.load(filename)
    return state
