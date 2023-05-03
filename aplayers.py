# TODO: currently all layers are still weight perturbed, not activity perturbed

import torch.nn as nn
import torch
import torch.nn.functional as F

class APertLinear(nn.Module):
    @torch.no_grad()
    def __init__(self, out_features: int, in_features: int) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(in_features, out_features))
        self.b = nn.Parameter(torch.randn(out_features))
        self._w = torch.randn(in_features, out_features)
        self._b = torch.randn(out_features)
        self.x = None

    @torch.no_grad()
    def forward(self, x, _x):
        self._w = torch.randn_like(self._w).normal_()
        self._b = torch.randn_like(self._b).normal_()
        y = torch.matmul(x, self.w) + self.b
        _y = torch.matmul(x, self._w) + torch.matmul(_x, self.w) + self._b
        return y, _y

    @torch.no_grad()
    def perturb(self, _objective):
        self.w.grad = self._w * _objective.mean().item()
        self.b.grad = self._b * _objective.mean().item()

class APertLRLinear(nn.Module):
    @torch.no_grad()
    def __init__(self, out_features: int, in_features: int) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(in_features, out_features))
        self.b = nn.Parameter(torch.randn(out_features))
        self._w = torch.randn(in_features, out_features)
        self._b = torch.randn(out_features)

    @torch.no_grad()
    def forward(self, x, _x):
        self._w = (torch.randn(self._w.shape[0], 1, device=self.w.device) * torch.randn(self._w.shape[1], device=self.w.device)).normal_()
        self._b = torch.randn_like(self._b, device=self.b.device).normal_()
        y = torch.matmul(x, self.w) + self.b
        _y = torch.matmul(x, self._w) + torch.matmul(_x, self.w) + self._b
        return y, _y

    @torch.no_grad()
    def perturb(self, _objective):
        self.w.grad = self._w * _objective.mean().item()
        self.b.grad = self._b * _objective.mean().item()

class APertReLU(nn.Module):
    @torch.no_grad()
    def __init__(self) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, x, _x):
        return x.relu() * 0.9 + x * 0.1, ((x >= 0) * 0.9 + 0.1) * _x

class APertConv2d(nn.Module):
    @torch.no_grad()
    def __init__(self, kernel_size: int, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        if type(kernel_size) is int:
            kernel_size = (kernel_size, kernel_size)
        self.kernel = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self._kernel = torch.randn_like(self.kernel)
        self._bias = torch.randn_like(self.bias)
        self.stride = stride

    @torch.no_grad()
    def forward(self, x, _x):
        self._kernel = torch.rand_like(self._kernel, device=self.kernel.device).normal_()
        self._bias = torch.rand_like(self._bias, device=self.bias.device).normal_()
        y = F.conv2d(x, self.kernel, self.bias, stride=self.stride)
        _y = F.conv2d(_x, self.kernel, stride=self.stride) + F.conv2d(x, self._kernel, stride=self.stride) + self._bias.reshape(-1, 1, 1)
        return y, _y

    @torch.no_grad()
    def perturb(self, _objective):
        self.kernel.grad = self._kernel * _objective.mean().item()
        self.bias.grad = self._bias * _objective.mean().item()

class APertLRConv2d(nn.Module):
    @torch.no_grad()
    def __init__(self, kernel_size: int, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        if type(kernel_size) is int:
            kernel_size = (kernel_size, kernel_size)
        self.kernel = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self._kernel = torch.randn_like(self.kernel)
        self._bias = torch.randn_like(self.bias)
        self.stride = stride

    @torch.no_grad()
    def forward(self, x, _x):
        self._kernel = (torch.rand(self._kernel.shape[0], self._kernel.shape[1], 1, 1, device=self.kernel.device) * 
                        torch.rand(self._kernel.shape[2], self._kernel.shape[3], device=self.kernel.device)).normal_()
        self._bias = torch.rand_like(self._bias, device=self.bias.device).normal_()
        y = F.conv2d(x, self.kernel, self.bias, stride=self.stride)
        _y = F.conv2d(_x, self.kernel, stride=self.stride) + F.conv2d(x, self._kernel, stride=self.stride) + self._bias.reshape(-1, 1, 1)
        return y, _y

    @torch.no_grad()
    def perturb(self, _objective):
        self.kernel.grad = self._kernel * _objective.mean().item()
        self.bias.grad = self._bias * _objective.mean().item()

class APertSoftmax(nn.Module):
    @torch.no_grad()
    def __init__(self) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, x, _x):
        y = x.softmax(dim=-1)
        return y, _x * y - torch.einsum("...i, ...i, ...j -> ...j", _x, y, y)

class APertAvgPool2d(nn.Module):
    @torch.no_grad()
    def __init__(self, kernel_size) -> None:
        super().__init__()
        self.kernel_size = kernel_size
    
    @torch.no_grad()
    def forward(self, x, _x):
        return F.avg_pool2d(x, self.kernel_size), F.avg_pool2d(_x, self.kernel_size)

class APertTanh(nn.Module):
    @torch.no_grad()
    def __init__(self) -> None:
        super().__init__()
    @torch.no_grad()
    def forward(self, x, _x):
        return torch.tanh(x), (1 - torch.tanh(x)**2) * _x

class APertLog(nn.Module):
    @torch.no_grad()
    def __init__(self) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, x, _x):
        return x.log(), _x / x