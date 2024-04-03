import torch


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PIN_MEMORY = True if DEVICE == 'cuda' else False
