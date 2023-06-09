from wplayers import *

class WPLeNet5(torch.nn.Module):
    def __init__(self, n_classes=10) -> None:
        super().__init__()
        self.conv_1 = WPertConv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.relu_1 = WPertReLU()
        self.pool_1 = WPertAvgPool2d(kernel_size=2)
        self.conv_2 = WPertConv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.relu_2 = WPertReLU()
        self.pool_2 = WPertAvgPool2d(kernel_size=2)
        self.conv_3 = WPertConv2d(in_channels=16, out_channels=120, kernel_size=4, stride=1)
        self.relu_3 = WPertReLU()
        self.flat_3 = nn.Flatten(-3, -1)
        self.linr_4 = WPertLRLinear(in_features=120, out_features=84)
        self.tanh_4 = WPertTanh()
        self.linr_5 = WPertLRLinear(in_features=84, out_features=n_classes)
        self.soft_5 = WPertSoftmax()
        self.loge_5 = WPertLog()

    def train(self, x, y):
        _x = torch.zeros_like(x)
        modu_list = [
            self.conv_1, self.relu_1, self.pool_1, 
            self.conv_2, self.relu_2, self.pool_2,
            self.conv_3, self.relu_3, lambda a,b: (self.flat_3(a), self.flat_3(b)),
            self.linr_4, self.tanh_4, 
            self.linr_5, self.soft_5, self.loge_5
        ]
        pert_list = [
            self.conv_1, self.conv_2, self.conv_3,
            self.linr_4, self.linr_5
        ]
        for modu in modu_list:
            x, _x = modu(x, _x)
            assert _x.shape == x.shape
        _objective = -_x[torch.arange(0, y.shape[-1]), y]
        for pert in pert_list:
            pert.perturb((_objective > 0) * 1.00)

    def predict(self, x):
        _x = torch.zeros_like(x)
        modu_list = [
            self.conv_1, self.relu_1, self.pool_1, 
            self.conv_2, self.relu_2, self.pool_2,
            self.conv_3, self.relu_3, lambda a,b: (self.flat_3(a), self.flat_3(b)),
            self.linr_4, self.tanh_4, 
            self.linr_5, self.soft_5, self.loge_5
        ]
        for modu in modu_list:
            x, _x = modu(x, _x)
        return torch.argmax(x, dim=-1)

class WPLRLeNet5(torch.nn.Module):
    def __init__(self, n_classes=10) -> None:
        super().__init__()
        self.conv_1 = WPertLRConv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.relu_1 = WPertReLU()
        self.pool_1 = WPertAvgPool2d(kernel_size=2)
        self.conv_2 = WPertLRConv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.relu_2 = WPertReLU()
        self.pool_2 = WPertAvgPool2d(kernel_size=2)
        self.conv_3 = WPertLRConv2d(in_channels=16, out_channels=120, kernel_size=4, stride=1)
        self.relu_3 = WPertReLU()
        self.flat_3 = nn.Flatten(-3, -1)
        self.linr_4 = WPertLRLinear(in_features=120, out_features=84)
        self.tanh_4 = WPertTanh()
        self.linr_5 = WPertLRLinear(in_features=84, out_features=n_classes)
        self.soft_5 = WPertSoftmax()
        self.loge_5 = WPertLog()

    def train(self, x, y):
        _x = torch.zeros_like(x)
        modu_list = [
            self.conv_1, self.relu_1, self.pool_1, 
            self.conv_2, self.relu_2, self.pool_2,
            self.conv_3, self.relu_3, lambda a,b: (self.flat_3(a), self.flat_3(b)),
            self.linr_4, self.tanh_4, 
            self.linr_5, self.soft_5, self.loge_5
        ]
        pert_list = [
            self.conv_1, self.conv_2, self.conv_3,
            self.linr_4, self.linr_5
        ]
        for modu in modu_list:
            x, _x = modu(x, _x)
        _objective = -_x[..., y]
        for pert in pert_list:
            pert.perturb((_objective > 0) * 1.00)

    def predict(self, x):
        _x = torch.zeros_like(x)
        modu_list = [
            self.conv_1, self.relu_1, self.pool_1, 
            self.conv_2, self.relu_2, self.pool_2,
            self.conv_3, self.relu_3, lambda a,b: (self.flat_3(a), self.flat_3(b)),
            self.linr_4, self.tanh_4, 
            self.linr_5, self.soft_5, self.loge_5
        ]
        for modu in modu_list:
            x, _x = modu(x, _x)
        return torch.argmax(x, dim=-1)

def local_loss(posx: torch.Tensor, negx: torch.Tensor):
    posx = posx.reshape(posx.shape[0], -1)
    posx = posx - posx.mean(dim=-1, keepdim=True)
    negx = negx.reshape(negx.shape[0], -1)
    negx = negx - negx.mean(dim=-1, keepdim=True)
    return ((negx**2 - posx**2 + 1).relu() + 1e-12).log().mean().exp()

class LocalMixerLeNet5(torch.nn.Module):
    def __init__(self, n_classes=10) -> None:
        super().__init__()
        self.conv_1 = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )
        self.conv_2 = torch.nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )
        self.conv_3 = torch.nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Flatten(-3, -1)
        )
        self.linr_4 = torch.nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh()
        )
        self.outp_5 = torch.nn.Sequential(
            nn.Linear(in_features=84, out_features=n_classes),
            nn.LogSoftmax(dim=-1)
        )
        self.last_x = nn.Parameter(torch.rand((1, 1, 28, 28)))

    def autograd_local_loss(self, x, y):
        px = x
        nx = torch.concat([self.last_x, x], dim=0)[torch.randperm(x.shape[0]+self.last_x.shape[0])][:x.shape[0]]
        self.last_x = nn.Parameter(x)
        for layer in [self.conv_1, self.conv_2, self.conv_3, self.linr_4]:
            ms = (torch.rand_like(px) >= 0.5) * 1.0
            # FIXME: construct negative sample with local mixer policy as the one in Hinton's paper is. 
            local_loss(layer(px), layer(px*ms + nx*(1-ms))).backward()
            with torch.no_grad():
                px = layer(px)
                nx = layer(nx)
        (-self.outp_5(px)[torch.arange(y.shape[0]), y]).mean().backward()

    @torch.no_grad()
    def predict(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.linr_4(x)
        x = self.outp_5(x)
        return x.argmax(dim=-1)