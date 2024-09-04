import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce


class VGG(nn.Module):

    def __init__(self, features, num_classes=1, init_weights=True):
        super(VGG, self).__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.3),
        )
        self.fc = nn.Linear(4096, num_classes)
        self.flat_0 = nn.Flatten()
        self.flat_1 = nn.Flatten()
        if init_weights:
            self._initialize_weights()

    def forward(self, img):

        x0 = self.features(img[:, 0, :, :, :])  # dwi
        x1 = self.features(img[:, 1, :, :, :])  # flair
        x0 = self.classifier(x0)
        x1 = self.classifier(x1)
        x0 = self.flat_0(x0)
        x1 = self.flat_1(x1)
        x_img = torch.cat((x0, x1), 1)
        x_img = self.fc(x_img)

        return x_img, x0, x1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _get_element_count(self, x):
        """
        Get the total number of elements in a layer output (not consider the batch level)
        """
        x_dim = list(x.size())[1:]
        return reduce(lambda x, y: x * y, x_dim)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool3d(kernel_size=2, stride=2)]
        else:
            conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv3d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)