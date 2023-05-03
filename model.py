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
            pert.perturb(_objective)

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
            pert.perturb(_objective)

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