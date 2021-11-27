import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from activations import Swish, Mish

act_fn_map = {
    'swish': 'silu'
}

memory_efficient_map = {
    'swish': Swish,
    'mish': Mish
}


def get_act_fn(act_fn, prefer_memory_efficient=True):
    if isinstance(act_fn, str):
        if prefer_memory_efficient and act_fn in memory_efficient_map:
            return memory_efficient_map[act_fn]()
        if act_fn in act_fn_map:
            act_fn = act_fn_map[act_fn]
        return getattr(F, act_fn)
    return act_fn


def drop_connect(x, drop_ratio, training=True):
    if not training or drop_ratio == 0:
        return x
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty(x.size(0), 1, 1, 1, device=x.device).bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


def print_num_params(model, display_all_modules=False):
    total_num_params = 0
    for n, p in model.named_parameters():
        num_params = 1
        for s in p.shape:
            num_params *= s
        if display_all_modules:
            print("{}: {}".format(n, num_params))
        total_num_params += num_params
    print("-" * 50)
    print("Total number of parameters: {}".format(total_num_params))


def weight_init(m):
    '''
    Usage:
        model.apply(weight_init)
    '''
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        init.normal_(m.weight)
        if m.bias is not None:
            init.normal_(m.bias)
    elif isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, (nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell)):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param)
            else:
                init.normal_(param)
