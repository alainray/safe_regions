# -*- coding: utf-8 -*-
import os.path

import torch
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
import torch.nn as nn
import numpy as np
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchvision.transforms import ToTensor
from os import mkdir
from os.path import join
seed = 123
torch.manual_seed(seed)

# dataset "MNIST", "CIFAR10"

dataset = "mnist"
data = {'mnist': MNIST, 'cifar10': CIFAR10, 'cifar100': CIFAR100}
train_data = data[dataset](root=".", download=True, train=True, transform=ToTensor())
test_data = data[dataset](root=".", download=True, train=False, transform=ToTensor())
# "fc", "minialex", "resnet"
arch = "fc"


# For a new value newValue, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far


class OnlineCovariance:
    """
    A class to calculate the mean and the covariance matrix
    of the incrementally added, n-dimensional data.
    """

    def __init__(self, order):
        """
        Parameters
        ----------
        order: int, The order (=="number of features") of the incrementally added
        dataset and of the resulting covariance matrix.
        """
        self._order = order
        self._shape = (order, order)
        self._identity = np.identity(order)
        self._ones = np.ones(order)
        self._count = 0
        self._mean = np.zeros(order)
        self._cov = np.zeros(self._shape)

    @property
    def count(self):
        """
        int, The number of observations that has been added
        to this instance of OnlineCovariance.
        """
        return self._count

    @property
    def mean(self):
        """
        double, The mean of the added data.
        """
        return self._mean

    @property
    def cov(self):
        """
        array_like, The covariance matrix of the added data.
        """
        return self._cov

    @property
    def corrcoef(self):
        """
        array_like, The normalized covariance matrix of the added data.
        Consists of the Pearson Correlation Coefficients of the data's features.
        """
        if self._count < 1:
            return None
        variances = np.diagonal(self._cov)
        denomiator = np.sqrt(variances[np.newaxis, :] * variances[:, np.newaxis])
        return self._cov / denomiator

    def add(self, observation):
        """
        Add the given observation to this object.

        Parameters
        ----------
        observation: array_like, The observation to add.
        """
        if self._order != len(observation):
            raise ValueError(f'Observation to add must be of size {self._order}')

        self._count += 1
        delta_at_nMin1 = np.array(observation - self._mean)
        self._mean += delta_at_nMin1 / self._count
        weighted_delta_at_n = np.array(observation - self._mean) / self._count
        shp = (self._order, self._order)
        D_at_n = np.broadcast_to(weighted_delta_at_n, self._shape).T
        D = (delta_at_nMin1 * self._identity).dot(D_at_n.T)
        self._cov = self._cov * (self._count - 1) / self._count + D

    def merge(self, other):
        """
        Merges the current object and the given other object into a new OnlineCovariance object.

        Parameters
        ----------
        other: OnlineCovariance, The other OnlineCovariance to merge this object with.

        Returns
        -------
        OnlineCovariance
        """
        if other._order != self._order:
            raise ValueError(
                f'''
                   Cannot merge two OnlineCovariances with different orders.
                   ({self._order} != {other._order})
                   ''')

        merged_cov = OnlineCovariance(self._order)
        merged_cov._count = self.count + other.count
        count_corr = (other.count * self.count) / merged_cov._count
        merged_cov._mean = (self.mean / other.count + other.mean / self.count) * count_corr
        flat_mean_diff = self._mean - other._mean
        shp = (self._order, self._order)
        mean_diffs = np.broadcast_to(flat_mean_diff, self._shape).T
        merged_cov._cov = (self._cov * self.count \
                           + other._cov * other._count \
                           + mean_diffs * mean_diffs.T * count_corr) \
                          / merged_cov.count
        return merged_cov


class FC(nn.Module):
    def __init__(self, input_dim=784, layer_sizes=[128, 256], n_classes=10, act=nn.ReLU):
        super(FC, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, layer_sizes[0]))
        self.act = act()
        for i in range(len(layer_sizes) - 1):
            layer_size = layer_sizes[i]
            next_layer = layer_sizes[i + 1]
            layers.append(nn.Linear(layer_size, next_layer))

        # classifier
        cls = nn.Linear(layer_sizes[-1], n_classes)
        layers.append(cls)
        print(layers)
        self.layers = nn.ModuleList(layers)
        print(self.layers)
        self.stats = []
        for i in range(len(layer_sizes)):
            self.stats.append(OnlineCovariance(layer_sizes[i]))
        self.stats.append(OnlineCovariance(n_classes))

    def forward(self, x):
        f = nn.Flatten()
        x = f(x)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            for rep in x:
                self.stats[i].add(rep.detach().cpu().numpy())
            if i < len(self.layers) - 1:
                x = self.act(x)

        return x


def checkpoint(model, stats, epoch, root="results", split="train"):
    if not os.path.isdir(root):
        mkdir(root)
    torch.save(model.state_dict, f"{root}/{arch}_{dataset}_{split}_{epoch}.pth")
    torch.save(model.stats, f"{root}/{arch}_{dataset}_{split}_{epoch}_stats.pth")
    torch.save(stats, f"{root}/{arch}_{dataset}_{split}_{epoch}_metrics.pth")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(model, loader, opt, device, criterion):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    model.train()
    for (x, label) in loader:
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

        print(f"\rLoss (Current): {loss_meter.val:.3f} Cum. Loss: {loss_meter.avg:.3f} \
        Acc: {100 * correct.float() / bs:.1f}% Cum. Acc: {100 * acc_meter.avg:.1f}%", end="")

    return model, [loss_meter, acc_meter]


def test(model, loader, device, criterion):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    model.eval()
    with torch.no_grad():
        for (x, label) in loader:
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

            print(f"\rLoss (Current): {loss_meter.val:.3f} Cum. Loss: {loss_meter.avg:.3f} \
            Acc: {100 * correct.float() / bs:.1f}% Cum. Acc: {100 * acc_meter.avg:.1f}%", end="")

    return model, [loss_meter, acc_meter]


models = {'fc': FC}  # ,'resnet':Resnet18, "minialex": MiniAlex}
optimizers = {'adam': Adam, 'sgd': SGD}
input_dims = {'mnist': 784, 'cifar10': 3 * 1024, 'cifar100': 3 * 1024}
optimizer = "adam"  # "sgd", "adam"
device = 'cuda'
criterion = CrossEntropyLoss()
lr = 0.001
model = models[arch](input_dim=input_dims[dataset])
opt = optimizers[optimizer](model.parameters(), lr=lr)
n_epochs = 10
train_dl = DataLoader(train_data, batch_size=128)
test_dl = DataLoader(test_data)

model.to(device)
for epoch in range(1, n_epochs + 1):
    print(f"\nTrain Epoch {epoch}")
    model, stats = train(model, train_dl, opt, device, criterion)
    checkpoint(model, stats, epoch, split="train")
    print(f"\nTest Epoch {epoch}")
    model, stats = test(model, train_dl, device, criterion)
    checkpoint(model, stats, epoch, split="test")
    # create checkpoint
    # record stats
    # record activation stats
    # record model.stats


