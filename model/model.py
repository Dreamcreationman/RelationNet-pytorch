import torch
from torch import nn
from torch.nn.functional import relu


class EmbeddingBlock(nn.Module):

    def __init__(self, in_channels=3, num_features=64):
        super(EmbeddingBlock, self).__init__()
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, 1, 1),
            nn.BatchNorm2d(num_features),
            nn.ReLU()
        )
        self.maxPooling1 = nn.MaxPool2d(2, padding=1)
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.BatchNorm2d(num_features),
            nn.ReLU()
        )
        self.maxPooling2 = nn.MaxPool2d(2, padding=1)
        self.convBlock3 = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.BatchNorm2d(num_features),
            nn.ReLU()
        )
        self.convBlock4 = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.BatchNorm2d(num_features),
            nn.ReLU()
        )

    def forward(self, x):
        # x: [n_way * k_shot, 3, 28, 28]
        x = self.maxPooling1(self.convBlock1(x))
        x = self.maxPooling2(self.convBlock2(x))
        out = self.convBlock4(self.convBlock3(x))
        return out # out: [n_way * k_shot, 64, 8, 8]


class RelationBlock(nn.Module):

    def __init__(self, num_features=64, fc_dim=8):
        super(RelationBlock, self).__init__()

        self.convBlock1 = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, 3, 1, 1),
            nn.BatchNorm2d(num_features),
            nn.ReLU()
        )
        self.maxPooling1 = nn.MaxPool2d(2, padding=1)
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.BatchNorm2d(num_features),
            nn.ReLU()
        )
        self.maxPooling2 = nn.MaxPool2d(2, padding=1)
        self.fc1 = nn.Linear(num_features * 3 * 3, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 1)

    def forward(self, x):
        # x: [num_class*num_per_class*num_class x 2*num_features x 8 x 8]
        x = self.maxPooling1(self.convBlock1(x))
        x = self.maxPooling2(self.convBlock2(x)) # [num_class*num_per_class*num_class x num_features x 3 x 3]
        out = x.view(x.size()[0], -1)
        out = relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out