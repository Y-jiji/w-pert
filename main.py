from torch.utils.data import dataloader
from torchvision.datasets import MNIST
from torchvision.transforms import *
from model import *
from torch.optim import *

device = 'cuda:0'

if __name__ == '__main__':
    # model = WPLeNet5(10).to(device).to(torch.double)
    # optim = Adam(model.parameters())
    # trans = Compose([ToTensor(), (lambda x: x.to(device).to(torch.double))])
    # rolling_acc = 0.1
    # rolling_dam = 0.999
    # for epoch in range(1000):
    #     print(f'------- epoch {epoch:>4} -------')
    #     # training
    #     trloader = dataloader.DataLoader(MNIST('./data', True,  transform=trans, download=True), 8, shuffle=False)
    #     for i, (image, label) in enumerate(trloader):
    #         model.zero_grad()
    #         model.train(image, label)
    #         optim.step()
    #         updater_acc = ((model.predict(image) == label.to(device)) * 1.0).mean().item();
    #         rolling_acc = rolling_acc * rolling_dam + updater_acc * (1-rolling_dam)
    #         print(rolling_acc, end='\r')
    #     print(f"{rolling_acc:<25} (rolling acc)")
    #     avg = []
    #     # compute testing accuracy
    #     tsloader = dataloader.DataLoader(MNIST('./data', False, transform=trans, download=True), 256, shuffle=True)
    #     for i, (image, label) in enumerate(tsloader):
    #         avg.append(((model.predict(image) == label.to(device)) * 1.0).mean().item())
    #     print(f"{sum(avg)/len(avg):<25} (testing acc)")
    model = LocalMixerLeNet5(10).to(device).to(torch.double)
    optim = AdamW(model.parameters())
    trans = Compose([ToTensor(), (lambda x: x.to(device).to(torch.double))])
    rolling_acc = 0.1
    rolling_dam = 0.999
    for epoch in range(1000):
        print(f'------- epoch {epoch:>4} -------')
        # training
        trloader = dataloader.DataLoader(MNIST('./data', True,  transform=trans, download=True), 8, shuffle=True)
        for i, (image, label) in enumerate(trloader):
            model.zero_grad()
            model.autograd_local_loss(image, label)
            optim.step()
            updater_acc = ((model.predict(image) == label.to(device)) * 1.0).mean().item();
            rolling_acc = rolling_acc * rolling_dam + updater_acc * (1-rolling_dam)
            print(rolling_acc, end='\r')
        print(f"{rolling_acc:<25} (rolling acc)")
        avg = []
        # compute testing accuracy
        tsloader = dataloader.DataLoader(MNIST('./data', False, transform=trans, download=True), 256, shuffle=True)
        for i, (image, label) in enumerate(tsloader):
            avg.append(((model.predict(image) == label.to(device)) * 1.0).mean().item())
        print(f"{sum(avg)/len(avg):<25} (testing acc)")