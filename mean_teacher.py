import torch
from itertools import chain


def update_ema_variables(model, ema_model, alpha):
    with torch.no_grad():
        # Use the true average until the exponential average is more correct
        # alpha = min(1 - 1 / (global_step + 1), alpha)

        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(other=param.data, alpha=1. - alpha)

        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            if torch.is_floating_point(ema_buffer):
                # ema_buffer.data.mul_(alpha).add_(1. - alpha, buffer.data)
                ema_buffer.data = buffer.data
            else:
                ema_buffer.data = buffer.data

def detach_model_parameters(model):
    for param in model.parameters():
        param.detach_()
