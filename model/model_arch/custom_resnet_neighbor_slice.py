import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from model.model_arch.attention.grid_attention_layer import GridAttentionBlock2D_TORR as AttentionBlock2D
from model.model_arch.attention.cbam import CBAM, ChannelAttention
from model.model_arch.attention.non_local_simple import NLBlockND
import timm

# from torchsummary import summary

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None, reduction_ratio=1, kernel_cbam=3, use_cbam=False):
        super(BasicBlock, self).__init__()
        self.use_cbam = use_cbam
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        if self.use_cbam:
            self.cbam = CBAM(n_channels_in=self.expansion * planes, reduction_ratio=reduction_ratio,
                             kernel_size=kernel_cbam)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_cbam:
            out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction_ratio=1, kernel_cbam=3, use_cbam=False):
        super(Bottleneck, self).__init__()
        self.use_cbam = use_cbam
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        if self.use_cbam:
            self.cbam = CBAM(n_channels_in=self.expansion * planes, reduction_ratio=reduction_ratio,
                             kernel_size=kernel_cbam)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.use_cbam:
            out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1, zero_init_residual=False, init_weights=True, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, reduction_ratio=1, kernel_cbam=3,
                 use_cbam_block=False, use_cbam_class=False, norm_layer=None, non_local=False, siamese=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.reduction_ratio = reduction_ratio
        self.kernel_cbam = kernel_cbam
        self.use_cbam_block = use_cbam_block
        self.use_cbam_class = use_cbam_class
        self.siamese = siamese

        # print(use_cbam_block, use_cbam_class)

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], non_local=non_local)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.attn_1 = timm.models.vision_transformer.Attention(
            dim=512, num_heads=4, qkv_bias=True)
        self.attn_2 = timm.models.vision_transformer.Attention(
            dim=512, num_heads=4, qkv_bias=True)
        self.attn_3 = timm.models.vision_transformer.Attention(
            dim=512, num_heads=4, qkv_bias=True)
        self.attn_4 = timm.models.vision_transformer.Attention(
            dim=512, num_heads=4, qkv_bias=True)
        self.attn_5 = timm.models.vision_transformer.Attention(
            dim=512, num_heads=4, qkv_bias=True)

        if self.use_cbam_class:
            self.cbam = ChannelAttention(n_channels_in=512 * block.expansion, reduction_ratio=reduction_ratio)
            self.cbam_class = nn.ModuleList()

        self.ind_features = nn.ModuleList()
        self.avgpool = nn.ModuleList()

        self.shared_features = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, self.layer1)
        for i in range(0, 5):
            self.ind_features.append(nn.Sequential(self.layer2, self.layer3, self.layer4))
            if self.use_cbam_class:
                self.cbam_class.append(self.cbam)
            self.avgpool.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.fc_0_1 = nn.Linear(512 * block.expansion * 5, 1)
        self.fc_1_1 = nn.Linear(512 * block.expansion * 5, 1)
        self.fc_2_1 = nn.Linear(512 * block.expansion * 5, 1)
        self.fc_3_1 = nn.Linear(512 * block.expansion * 5, 1)
        self.fc_4_1 = nn.Linear(512 * block.expansion * 5, 1)
        self.fc_0_2 = nn.Linear(512 * block.expansion, 1)

        self.classifier = nn.Sequential(
            nn.Softmax(dim=2),
            nn.Linear(25, num_classes)
        )
        self.fc = nn.Linear(512*25, num_classes)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, non_local=False, dilate=False):
        strides = [stride] + [1] * (blocks - 1)
        last_idx = len(strides)
        if non_local:
            last_idx = len(strides) - 1
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, self.reduction_ratio, self.kernel_cbam,
                            self.use_cbam_block))
        self.inplanes = planes * block.expansion
        for _ in range(1, last_idx):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, reduction_ratio=self.reduction_ratio,
                                kernel_cbam=self.kernel_cbam, use_cbam=self.use_cbam_block))
        if non_local:
            layers.append(NLBlockND(in_channels=self.inplanes, dimension=2))
            layers.append(block(self.inplanes, planes, stride=strides[-1]))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        batch_size, _, z_dim, _, _ = x.shape
        temp_0, temp_1, temp_2, temp_3, temp_4 = ([] for i in range(5))
        # if single modal input
        # ADC or NCCT
        # x = torch.stack([x[:, 0, :, :, :], x[:, 0, :, :, :]], 1)
        # DWI or CTA
        # x = torch.stack([x[:, 1, :, :, :], x[:, 1, :, :, :]], 1)
        # FLAIR
        # x = torch.stack([x[:, 2, :, :, :], x[:, 2, :, :, :], x[:, 2, :, :, :]], 1)
        for i in range(0, 5):
            f = self.shared_features(x[:, :, i, :, :])
            if self.use_cbam_class:
                f = f + self.cbam_class[i](f)
            f = self.ind_features[0](f)
            f = self.avgpool[0](f)
            f = nn.functional.dropout(f, p=0.2)
            f = torch.flatten(f, 1)
            f = f[None, ...]
            temp_0.append(f)
        temp_0 = torch.squeeze(torch.stack(temp_0, 1), 0).permute(1, 0, 2)
        temp_0 = temp_0 + self.attn_1(temp_0)

        for i in range(5, 10):
            f = self.shared_features(x[:, :, i, :, :])
            if self.use_cbam_class:
                f = f + self.cbam_class[i](f)
            f = self.ind_features[1](f)
            f = self.avgpool[1](f)
            f = nn.functional.dropout(f, p=0.2)
            f = torch.flatten(f, 1)
            f = f[None, ...]
            temp_1.append(f)
        temp_1 = torch.squeeze(torch.stack(temp_1, 1), 0).permute(1, 0, 2)
        temp_1 = temp_1 + self.attn_1(temp_1)

        for i in range(10, 15):
            f = self.shared_features(x[:, :, i, :, :])
            if self.use_cbam_class:
                f = f + self.cbam_class[i](f)
            f = self.ind_features[2](f)
            f = self.avgpool[2](f)
            f = nn.functional.dropout(f, p=0.2)
            f = torch.flatten(f, 1)
            f = f[None, ...]
            temp_2.append(f)
        temp_2 = torch.squeeze(torch.stack(temp_2, 1), 0).permute(1, 0, 2)
        temp_2 = temp_2 + self.attn_1(temp_2)

        for i in range(15, 20):
            f = self.shared_features(x[:, :, i, :, :])
            if self.use_cbam_class:
                f = f + self.cbam_class[i](f)
            f = self.ind_features[3](f)
            f = self.avgpool[3](f)
            f = nn.functional.dropout(f, p=0.2)
            f = torch.flatten(f, 1)
            f = f[None, ...]
            temp_3.append(f)
        temp_3 = torch.squeeze(torch.stack(temp_3, 1), 0).permute(1, 0, 2)
        temp_3 = temp_3 + self.attn_1(temp_3)

        for i in range(20, 25):
            f = self.shared_features(x[:, :, i, :, :])
            if self.use_cbam_class:
                f = f + self.cbam_class[i](f)
            f = self.ind_features[4](f)
            f = self.avgpool[4](f)
            f = nn.functional.dropout(f, p=0.2)
            f = torch.flatten(f, 1)
            f = f[None, ...]
            temp_4.append(f)
        temp_4 = torch.squeeze(torch.stack(temp_4, 1), 0).permute(1, 0, 2)
        temp_4 = temp_4 + self.attn_1(temp_4)

        return temp_0, temp_1, temp_2, temp_3, temp_4

    def forward(self, x):
        if not self.siamese:
            x_all = []
            temp_0, temp_1, temp_2, temp_3, temp_4 = self._forward_impl(x)
            fc_0 = self.fc_0_1(temp_0.flatten(1))
            x_all.append(temp_0)
            fc_1 = self.fc_1_1(temp_1.flatten(1))
            x_all.append(temp_1)
            fc_2 = self.fc_2_1(temp_2.flatten(1))
            x_all.append(temp_2)
            fc_3 = self.fc_3_1(temp_3.flatten(1))
            x_all.append(temp_3)
            fc_4 = self.fc_4_1(temp_4.flatten(1))
            x_all.append(temp_4)
            x_all = torch.cat(x_all, dim=1)
            x_all = nn.functional.dropout(x_all, p=0.2)

            # when downstream task, do below
            # x_all = self.fc_0_2(x_all)
            # x_all = x_all.permute(2, 0, 1)
            # return torch.squeeze(self.classifier(x_all), 0), fc_0, fc_1, fc_2, fc_3, fc_4

            # when SSL:
            # return self.fc(x_all.flatten(1))

            x_all = self.fc_0_2(x_all)
            x_all = x_all.permute(2, 0, 1)
            return torch.squeeze(self.classifier(x_all), 0), fc_0, fc_1, fc_2, fc_3, fc_4

        else:
            left_input = x[:, :, :, :, int(x.shape[3] / 2):]
            right_input = torch.flip(x[:, :, :, :, int(x.shape[3] / 2):], [2])

            output1 = self._forward_impl(left_input)
            output1 = torch.cat(output1, dim=1)
            output2 = self._forward_impl(right_input)
            output2 = torch.cat(output2, dim=1)
            output = torch.abs(output1 - output2)
            output = self.fc_0_2(output).permute(2, 0, 1)

            return torch.squeeze(self.classifier(output), 0)


class ResNetGatedAtt(nn.Module):

    def __init__(self, block, layers, num_classes=1, zero_init_residual=False, init_weights=True,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, aggregation_mode='ft', single_att=False):
        super(ResNetGatedAtt, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.single_att = single_att

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=4, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.classifier_final = nn.Sequential(
            nn.Softmax(dim=2),
            nn.Linear(22, 1),
        )
        self.shared_features = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                             self.layer1)
        self.features_a = nn.ModuleList()
        self.features_b = nn.ModuleList()
        self.features_c = nn.ModuleList()
        self.avgpool = nn.ModuleList()
        self.attention_a = nn.ModuleList()
        self.attention_b = nn.ModuleList()
        for i in range(0, 5):
            self.features_a.append(self.layer2)
            self.features_b.append(self.layer3)
            self.features_c.append(self.layer4)
            self.avgpool.append(nn.AdaptiveAvgPool2d((1, 1)))
            self.attention_a.append(AttentionBlock2D(in_channels=128, gating_channels=512,
                                                     inter_channels=512, sub_sample_factor=(1, 1),
                                                     nonlinearity1='relu'))

            self.attention_b.append(AttentionBlock2D(in_channels=256, gating_channels=512,
                                                     inter_channels=512, sub_sample_factor=(1, 1),
                                                     nonlinearity1='relu'))
        if aggregation_mode == 'concat':

            if self.single_att:
                self.classifier = nn.Linear(256 + 512, num_classes)
            else:
                self.classifier = nn.Linear(128 + 256 + 512, num_classes)
            self.aggregate = self.aggregation_concat

        else:
            self.classifier0 = nn.Linear(128, num_classes)
            self.classifier1 = nn.Linear(256, num_classes)
            self.classifier2 = nn.Linear(512, num_classes)
            if self.single_att:
                self.classifiers = [self.classifier1, self.classifier2]
            else:
                self.classifiers = [self.classifier0, self.classifier1, self.classifier2]

            if aggregation_mode == 'mean':
                self.aggregate = self.aggregation_sep

            elif aggregation_mode == 'deep_sup':
                if self.single_att:
                    self.classifier = nn.Linear(256 + 512, num_classes)
                else:
                    self.classifier = nn.Linear(128 + 256 + 512, num_classes)
                self.aggregate = self.aggregation_ds

            elif aggregation_mode == 'ft':
                if self.single_att:
                    self.classifier = nn.Linear(num_classes * 2, num_classes)
                else:
                    self.classifier = nn.Linear(num_classes * 3, num_classes)
                self.aggregate = self.aggregation_ft
            else:
                raise NotImplementedError

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # print(self.aggregate)
        batch_size, _, z_dim, _, _ = x.shape
        x_all = []
        for i in range(2, 5):
            f = self.shared_features(x[:, :, i, :, :])
            fa = self.features_a[0](f)
            fb = self.features_b[0](fa)
            fc = self.features_c[0](fb)

            g_conv_a, att_a = self.attention_a[0](fa, fc)
            g_conv_b, att_b = self.attention_b[0](fb, fc)
            fc = self.avgpool[0](fc)
            fc = torch.flatten(fc, 1)
            # fc = nn.functional.dropout(fc, p=0.5)
            g_a = torch.sum(g_conv_a.view(batch_size, 128, -1), dim=-1)
            g_b = torch.sum(g_conv_b.view(batch_size, 256, -1), dim=-1)
            if self.single_att:
                single = self.aggregate(F.softmax(g_b), fc)
            else:
                single = self.aggregate(F.softmax(g_a), F.softmax(g_b), fc)
            x_all.append(single)
        for i in range(5, 10):
            f = self.shared_features(x[:, :, i, :, :])
            fa = self.features_a[1](f)
            fb = self.features_b[1](fa)
            fc = self.features_c[1](fb)

            g_conv_a, att_a = self.attention_a[1](fa, fc)
            g_conv_b, att_b = self.attention_b[1](fb, fc)
            # fc = nn.functional.dropout(fc, p=0.5)
            fc = self.avgpool[1](fc)
            fc = torch.flatten(fc, 1)
            g_a = torch.sum(g_conv_a.view(batch_size, 128, -1), dim=-1)
            g_b = torch.sum(g_conv_b.view(batch_size, 256, -1), dim=-1)
            if self.single_att:
                single = self.aggregate(F.softmax(g_b), fc)
            else:
                single = self.aggregate(F.softmax(g_a), F.softmax(g_b), fc)
            x_all.append(single)
        for i in range(10, 15):
            f = self.shared_features(x[:, :, i, :, :])
            fa = self.features_a[2](f)
            fb = self.features_b[2](fa)
            fc = self.features_c[2](fb)

            g_conv_a, att_a = self.attention_a[2](fa, fc)
            g_conv_b, att_b = self.attention_b[2](fb, fc)
            # fc = nn.functional.dropout(fc, p=0.5)
            fc = self.avgpool[2](fc)
            fc = torch.flatten(fc, 1)
            g_a = torch.sum(g_conv_a.view(batch_size, 128, -1), dim=-1)
            g_b = torch.sum(g_conv_b.view(batch_size, 256, -1), dim=-1)
            if self.single_att:
                single = self.aggregate(F.softmax(g_b), fc)
            else:
                single = self.aggregate(F.softmax(g_a), F.softmax(g_b), fc)
            x_all.append(single)
        for i in range(15, 20):
            f = self.shared_features(x[:, :, i, :, :])
            fa = self.features_a[3](f)
            fb = self.features_b[3](fa)
            fc = self.features_c[3](fb)

            g_conv_a, att_a = self.attention_a[3](fa, fc)
            g_conv_b, att_b = self.attention_b[3](fb, fc)
            # fc = nn.functional.dropout(fc, p=0.5)
            fc = self.avgpool[3](fc)
            fc = torch.flatten(fc, 1)

            g_a = torch.sum(g_conv_a.view(batch_size, 128, -1), dim=-1)
            g_b = torch.sum(g_conv_b.view(batch_size, 256, -1), dim=-1)
            if self.single_att:
                single = self.aggregate(F.softmax(g_b), fc)
            else:
                single = self.aggregate(F.softmax(g_a), F.softmax(g_b), fc)
            x_all.append(single)
        for i in range(20, 24):
            f = self.shared_features(x[:, :, i, :, :])
            fa = self.features_a[4](f)
            fb = self.features_b[4](fa)
            fc = self.features_c[4](fb)

            g_conv_a, att_a = self.attention_a[4](fa, fc)
            g_conv_b, att_b = self.attention_b[4](fb, fc)
            # fc = nn.functional.dropout(fc, p=0.5)
            fc = self.avgpool[4](fc)
            fc = torch.flatten(fc, 1)

            g_a = torch.sum(g_conv_a.view(batch_size, 128, -1), dim=-1)
            g_b = torch.sum(g_conv_b.view(batch_size, 256, -1), dim=-1)
            if self.single_att:
                single = self.aggregate(F.softmax(g_b), fc)
            else:
                single = self.aggregate(F.softmax(g_a), F.softmax(g_b), fc)
            x_all.append(single)

        x_all = torch.stack(x_all, dim=0)
        x_all = x_all.permute(2, 1, 0)

        return torch.squeeze(self.classifier_final(x_all), 0)

    def forward(self, x):
        return self._forward_impl(x)

    def aggregation_sep(self, *attended_maps):
        return [clf(att) for clf, att in zip(self.classifiers, attended_maps)]

    def aggregation_ft(self, *attended_maps):
        preds = self.aggregation_sep(*attended_maps)
        return self.classifier(torch.cat(preds, dim=1))

    def aggregation_ds(self, *attended_maps):
        preds_sep = self.aggregation_sep(*attended_maps)
        pred = self.aggregation_concat(*attended_maps)
        return [pred] + preds_sep

    def aggregation_concat(self, *attended_maps):
        return self.classifier(torch.cat(attended_maps, dim=1))


def _resnet(arch, block, layers, pretrained, progress, num_classes, zero_init_residual, att=False, model_path=None,
            **kwargs):
    if not att:
        model = ResNet(block, layers, num_classes=num_classes, zero_init_residual=zero_init_residual, **kwargs)
    elif att == 'gated':
        model = ResNetGatedAtt(block, layers, num_classes=num_classes, zero_init_residual=zero_init_residual,
                               aggregation_mode="ft", single_att=True, **kwargs)
    elif att == 'cbam':
        model = ResNet(block, layers, num_classes=num_classes, zero_init_residual=zero_init_residual,
                       reduction_ratio=1, kernel_cbam=3, use_cbam_block=True,
                       use_cbam_class=True, **kwargs)
    elif att == 'nonlocal':
        model = ResNet(block, layers, num_classes=num_classes, zero_init_residual=zero_init_residual,
                       reduction_ratio=1, non_local=True, **kwargs)
    # when use both cbam and nonlocal
    elif att == 'cn':
        model = ResNet(block, layers, num_classes=num_classes, zero_init_residual=zero_init_residual,
                       reduction_ratio=1, kernel_cbam=3, use_cbam_block=True,
                       use_cbam_class=False, non_local=True, **kwargs)
    # print(type(model))
    pretrained_fc = True
    # from torchsummary import summary
    # model.cuda()
    # summary(model.ind_features[0], (3, 26, 224, 224))

    if pretrained:
        kwargs['init_weights'] = False
        if model_path is None:
            print("please specify pretrained model path")
            sys.exit()
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['state_dict']
        if pretrained_fc:
            pretrained_dict = {k: v for k, v in state_dict.items() if
                               k in model.state_dict() and not k.startswith("classifier")}
        else:
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        # for param in model.parameters():
        #     param.requires_grad = False
        # model.classifier.requires_grad = True
        # for (name, module) in model.named_children():
        #     if name == 'cbam' or name == 'cbam_class':
        #         for layer in module.children():
        #             for param in layer.parameters():
        #                 param.requires_grad = True
        #     else:
        #         for layer in module.children():
        #             for i in layer.children():
        #                 if type(i) == CBAM:
        #                     for param in layer.parameters():
        #                         param.requires_grad = True

    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, num_classes=1, zero_init_residual=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   num_classes=num_classes, zero_init_residual=zero_init_residual, **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
