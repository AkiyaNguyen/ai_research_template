import torch

def get_proper_device(device_str=None):
    if torch.cuda.is_available():
        auto_device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        auto_device = torch.device('mps')
    else:
        auto_device = torch.device('cpu')

    if device_str is None:
        return auto_device

    device_str = device_str.lower().strip()

    if device_str == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    elif device_str == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')
    elif device_str == 'cpu':
        return torch.device('cpu')
    else:
        print(f"Warning: Requested device '{device_str}' is not available. Falling back to {auto_device}.")
        return auto_device
    