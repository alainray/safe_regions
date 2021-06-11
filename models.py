from utils import OnlineCovariance
import torch.nn as nn


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
