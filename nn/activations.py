import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryEfficientSwish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):
    def forward(self, x):
        return MemoryEfficientSwish.apply(x)


class MemoryEfficientMish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.tanh(F.softplus(i))
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        v = 1. + i.exp()
        h = v.log() 
        grad_gh = 1./h.cosh().pow_(2) 
        grad_hx = i.sigmoid()
        grad_gx = grad_gh *  grad_hx
        grad_f =  torch.tanh(F.softplus(i)) + i * grad_gx 
        return grad_output * grad_f 


class Mish(nn.Module):
    def forward(self, x):
        return MemoryEfficientMish.apply(x)
