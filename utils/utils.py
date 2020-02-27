import torch
import random
import os
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    except Exception as e:
        print('Failed to set cuda seed')

def calc_global_step(dataset_len, world_size, batch_size_per_worker, epoch):
    return int(dataset_len // (world_size * batch_size_per_worker) * epoch)

def get_trainable_params(model):
    trainable_params = []

    for param in model.parameters():
        if param.requires_grad:
            trainable_params.append(param)

    return trainable_params

def get_max_lr(optimizer):
    max_lr = 0.
    for i, param_group in enumerate(optimizer.param_groups):
        max_lr = max(float(param_group['lr']), max_lr)
    return max_lr

def freeze_bn(module):
    '''Freeze BatchNorm layers.'''
    for layer in [module for module in module.modules() if type(module) != torch.nn.Sequential]:
        if isinstance(layer, torch.nn.BatchNorm2d):
            layer.eval()
            print('frozen', layer)


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = torch.distributed.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp

def memory_report():
    t = torch.cuda.get_device_properties(0).total_memory
    c = torch.cuda.memory_cached(0)
    a = torch.cuda.memory_allocated(0)
    f = c - a  # free inside cache
    print('Mem:', t, c, a, f)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg
