import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import pdb


def _get_element_count(x):
    """
    Get the total number of elements in a layer output (not consider the batch level)
    """
    x_dim = list(x.size())[1:]
    return reduce(lambda x, y: x * y, x_dim)


class Alexnet_3D(nn.Module):
    """
    """

    def __init__(self, num_channels=8, num_units_fc3=84):
        super(Alexnet_3D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(num_channels, 20, kernel_size=(2, 3, 3), padding=1),
            nn.BatchNorm3d(20),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(20, 20, kernel_size=(2, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(20, 40, kernel_size=(2, 3, 3), padding=1),
            nn.BatchNorm3d(40),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),

        )

        self.fc1 = nn.Linear(5760, 120)
        self.fc2 = nn.Linear(120, num_units_fc3)
        self.fc3 = nn.Linear(num_units_fc3, 2)

    def forward(self, img, avg_probs):
        # x2 = volume.type(torch.cuda.FloatTensor)
        # x3 = area.type(torch.cuda.FloatTensor)
        # x2 = x2.view(x2.size(0),-1)
        # x3 = x3.view(x3.size(0),-1)
        x = self.features(img)
        x = x.view(-1, _get_element_count(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)

        if avg_probs.cpu().numpy().all() != 0:
            x4 = avg_probs.type(torch.cuda.FloatTensor)
            x4 = x4.view(x4.size(0), -1)
            x = torch.cat([x, x4], -1)

        x = F.relu(self.fc3(x))
        return F.softmax(x, dim=1)


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # nn.init.xavier_normal_(m.weight.data)
        # nn.init.xavier_uniform_(m.weight.data)
        nn.init.kaiming_normal_(m.weight.data)
        # nn.init.kaiming_uniform_(m.weight.data)
        # m.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)


def _get_element_count(x):
    """
    Get the total number of elements in a layer output (not consider the batch level)
    """
    x_dim = list(x.size())[1:]
    return reduce(lambda x, y: x * y, x_dim)


def subnetwork_share_weight():
    # Alexnet
    features = nn.Sequential(
        nn.Conv3d(3, 64, kernel_size=11, stride=4, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(kernel_size=2, stride=2),
        nn.BatchNorm3d(64),
        nn.Conv3d(64, 192, kernel_size=5, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(kernel_size=2, stride=2),
        nn.BatchNorm3d(192),
        nn.Conv3d(192, 384, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv3d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv3d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(kernel_size=1, stride=2),
    )
    features.apply(weight_init)
    return features


class Siamese_alexnet(nn.Module):

    def __init__(self, num_classes=1):
        super(Siamese_alexnet, self).__init__()
        self.share_weight_features = subnetwork_share_weight()
        self.flat_0 = nn.Flatten()
        self.flat_1 = nn.Flatten()
        self.fc1 = nn.Linear(25088, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, img):

        x0 = self.share_weight_features(img[:, 0, :, :, :, :])  # dwi
        x1 = self.share_weight_features(img[:, 1, :, :, :, :])  # flair
        x0 = self.flat_0(x0)
        x1 = self.flat_1(x1)
        x_img = torch.cat((x0, x1), 1)
        x_img = F.dropout(x_img, training=self.training)
        x_img = F.relu(self.fc1(x_img))
        x_img = F.dropout(x_img, training=self.training)
        x_img = F.relu(self.fc2(x_img))
        x_img = self.fc3(x_img)
        return x_img, x0, x1
