# -*- coding: utf-8 -*-
import torch
from models import FC, MiniAlexNet
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from utils import AverageMeter, checkpoint
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchvision.transforms import ToTensor

seed = 123
torch.manual_seed(seed)

# dataset "MNIST", "CIFAR10"

dataset = "cifar100"
data = {'mnist': MNIST, 'cifar10': CIFAR10, 'cifar100': CIFAR100}
train_data = data[dataset](root=".", download=True, train=True, transform=ToTensor())
test_data = data[dataset](root=".", download=True, train=False, transform=ToTensor())
# "fc", "minialex", "resnet"
arch = "minialex"


def train(model, loader, opt, device, criterion):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    model.train()
    total_batches = len(loader)
    for n_batch, (x, label) in enumerate(loader):
        opt.zero_grad()
        x = x.to(device)
        label = label.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct = (preds == label).cpu().sum()
        bs = x.shape[0]
        loss = criterion(logits, label)
        loss.backward()
        # Update stats
        loss_meter.update(loss.cpu(), bs)
        # print(correct,bs)
        acc_meter.update(correct.cpu() / float(bs), bs)
        opt.step()

        print(f"\r {n_batch+1}/{total_batches}: Loss (Current): {loss_meter.val:.3f} Cum. Loss: {loss_meter.avg:.3f} \
        Acc: {100 * correct.float() / bs:.1f}% Cum. Acc: {100 * acc_meter.avg:.1f}%", end="")

    return model, [loss_meter, acc_meter]


def test(model, loader, device, criterion):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    total_batches = len(loader)
    model.eval()
    with torch.no_grad():
        for n_batch, (x, label) in enumerate(loader):
            x = x.to(device)
            label = label.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct = (preds == label).cpu().sum()
            bs = x.shape[0]
            loss = criterion(logits, label)
            # Update stats
            loss_meter.update(loss.cpu(), bs)
            acc_meter.update(correct.cpu() / float(bs), bs)

            print(
            f"\r {n_batch + 1}/{total_batches}: Loss (Current): {loss_meter.val:.3f} Cum. Loss: {loss_meter.avg:.3f} \
            Acc: {100 * correct.float() / bs:.1f}% Cum. Acc: {100 * acc_meter.avg:.1f}%", end="")

    return model, [loss_meter, acc_meter]


in_channels = {'mnist': 1, 'cifar10': 3, 'cifar100': 3}
n_classes = {'mnist': 10, 'cifar10': 10, 'cifar100': 100}
models = {'fc': FC, "minialex": MiniAlexNet}  # ,'resnet':Resnet18}
optimizers = {'adam': Adam, 'sgd': SGD}
input_dims = {'mnist': 784, 'cifar10': 3 * 1024, 'cifar100': 3 * 1024}
optimizer = "adam"  # "sgd", "adam"
device = 'cuda'
criterion = CrossEntropyLoss()
lr = 0.001
if arch == 'fc':
    model = models[arch](input_dim=input_dims[dataset], n_classes=n_classes[dataset])
elif arch == 'minialex':
    model = models[arch](n_channels=in_channels[dataset], n_classes=n_classes[dataset])
else:
    raise ValueError
opt = optimizers[optimizer](model.parameters(), lr=lr)
n_epochs = 10
train_dl = DataLoader(train_data, batch_size=128)
test_dl = DataLoader(test_data, batch_size=512)

model.to(device)
for epoch in range(1, n_epochs + 1):
    print(f"\nTrain Epoch {epoch}")
    model, stats = train(model, train_dl, opt, device, criterion)
    checkpoint(model, stats, epoch, arch, dataset, split="train")
    print(f"\nTest Epoch {epoch}")
    model, stats = test(model, train_dl, device, criterion)
    checkpoint(model, stats, epoch, arch, dataset, split="test")
    # create checkpoint
    # record stats
    # record activation stats
    # record model.stats


