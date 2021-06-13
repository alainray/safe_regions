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


class MiniAlexNet(nn.Module):

    def __init__(self, n_classes=10, n_channels=3):
        super(MiniAlexNet, self).__init__()
        self.relu = nn.ReLU()
        self.f = nn.Flatten()
        self.conv1 = nn.Conv2d(n_channels, 200, kernel_size=5, stride=2, padding=1)
        #self.bn1 = nn.BatchNorm2d(200)
        #self.conv1 = nn.ReLU(inplace=True)
        self.mp = nn.MaxPool2d(kernel_size=3, stride=1)

        self.conv2 = nn.Conv2d(200, 200, kernel_size=5, stride=2, padding=1)
        #self.bn2 = nn.BatchNorm2d(200)
        #self.conv1 = nn.ReLU(inplace=True)
        #self.conv1 = nn.MaxPool2d(kernel_size=3, stride=1)


        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(200 * 6 * 6, 384)
        #self.bn3 = nn.BatchNorm1d(384)
            #nn.ReLU(inplace=True),

        self.fc2 = nn.Linear(384, 192)
        #self.bn4 = nn.BatchNorm1d(192)
        #nn.ReLU(inplace=True),

        self.cls = nn.Linear(192, n_classes)

        self.stats = []
        self.stats.append(OnlineCovariance(200))
        self.stats.append(OnlineCovariance(200))
        self.stats.append(OnlineCovariance(384))
        self.stats.append(OnlineCovariance(192))
        self.stats.append(OnlineCovariance(n_classes))

    def forward(self, x):
        def batch_stats(batch, n_counter, layer_type = 'fc'):
            if layer_type == 'fc':
                for rep in batch:
                    self.stats[n_counter].add(rep.detach().cpu().numpy())
            else:
                # dimensions are (batch, channels, h, w)
                # change to (batch, h, w, channels) with permute
                batch = batch.permute(0,2,3,1)
                batch = batch.mean(dim=[1,2])
                #*_, channels = batch.shape
                #batch = batch.reshape((-1, channels))
                # then keep
                for rep in batch:
                    self.stats[n_counter].add(rep.detach().cpu().numpy())

        x = self.conv1(x)
        batch_stats(x, 0, layer_type='conv')
        x = self.relu(x)
        x = self.mp(x)
        x = self.conv2(x)
        batch_stats(x, 1, layer_type='conv')
        x = self.relu(x)
        x = self.mp(x)
        x = self.avgpool(x)
        x = self.f(x)
        x = self.fc1(x)
        batch_stats(x, 2, layer_type='fc')
        x = self.relu(x)
        x = self.fc2(x)
        batch_stats(x, 3, layer_type='fc')
        x = self.relu(x)
        x = self.cls(x)
        batch_stats(x, 4, layer_type='fc')

        return x

