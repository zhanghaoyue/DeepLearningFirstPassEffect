"""
A list of models.
"""
import torch
import torchvision.models.video
from torch.cuda.amp import autocast
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.utils.model_zoo as model_zoo
import model.model_arch.resnet_m2net as rsb
import model.model_arch.custom_resnet_neighbor_slice as res25
import model.model_arch.custom_resnet as rs_slice
import model.model_arch.densenet as dn
import model.model_arch.alexnet as ax
import model.model_arch.resnet as rs
import model.model_arch.vgg as vgg
import model.model_arch.wide_resnet as wrs
import model.model_arch.Siamese_WResnet as swrs
import model.model_arch.siamese_unet3d as su
import utils
import model.model_arch.grid_attention_layer as atten
import pathlib
from collections import OrderedDict
import timm

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# structure of VGG
cfg = {
    'A': [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M'],
    """ 'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],"""
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def siamese_resnet18(pretrained=False, **kwargs):
    kwargs['init_weights'] = True
    model = rs.Siamese_Resnet(rs.BasicBlock, [2, 2, 2, 2], **kwargs)

    if pretrained:
        # kwargs['init_weights'] = False
        model_path = "/haoyuezhang/pretrained_models/resnet-18-kinetics.pth"
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['state_dict']
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not k.startswith("fc")}
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        ct = 0

        for child in model.children():
            ct += 1
            if ct < 4:
                for param in child.parameters():
                    param.requires_grad = False
    return model


def single_resnet(pretrained=False, **kwargs):
    kwargs['init_weights'] = True
    # set resnet 10 1111, 18 2222 50 3463
    model = rs.Single_Resnet(rs.BasicBlock, [2, 2, 2, 2], **kwargs)

    if pretrained:
        kwargs['init_weights'] = False
        model_path = "/haoyuezhang/pretrained_models/resnet-18-kinetics.pth"
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['state_dict']
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not k.startswith("fc")}
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        ct = 0

        for child in model.children():
            ct += 1
            if ct < 6:
                for param in child.parameters():
                    param.requires_grad = False

    return model


def siamese_resnet50(pretrained=False, **kwargs):
    kwargs['init_weights'] = True
    model = rs.Siamese_Resnet(rs.Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained:
        # kwargs['init_weights'] = False
        model_path = "/haoyuezhang/pretrained_models/resnet-50-kinetics.pth"
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['state_dict']
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not k.startswith("fc")}
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        ct = 0
        for child in model.children():
            ct += 1
            if ct < 7:
                for param in child.parameters():
                    param.requires_grad = False
    return model


def siamese_wresnet50(pretrained=False, **kwargs):
    if not pretrained:
        model = swrs.Siamese_WResnet(swrs.WideBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def siamese_wresnet18(pretrained=False, **kwargs):
    """Constructs a wide ResNet-50 model.
    """
    if not pretrained:
        model = wrs.Siamese_WResnet(wrs.WideBottleneck, [2, 2, 2, 2], **kwargs)
    return model


def siamese_alexnet(pretrained=False, **kwargs):
    model = ax.Siamese_alexnet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model


def m2net(pretrained=False, **kwargs):
    class M2Net(nn.Module):
        def __init__(self, base_model):
            super(M2Net, self).__init__()
            self.base_model_0 = base_model
            self.base_model_1 = base_model
            self.base_model_2 = base_model
            self.bl_pooling = rsb.Bilinear_Pooling_2(1)
            # Remove the pooling layer and full connection layer
            self.fc_1 = nn.Linear(1024, 64)
            self.fc_2 = nn.Linear(64, 1)
            self.fc_3 = nn.Linear(3, 1)
            self.fc_4 = nn.Linear(3, 1)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        def forward(self, x):
            features_0, fc_0 = self.base_model_0(x[:, 0, :, :, :])
            features_1, fc_1 = self.base_model_1(x[:, 1, :, :, :])
            features_2, fc_2 = self.base_model_2(x[:, 2, :, :, :])

            # bilinear pooling
            shared, fusion = self.bl_pooling(features_0, features_1, features_2,fc_0, fc_1, fc_2)

            return fusion, fc_0, fc_1, fc_2, shared

    base_model = rsb.resnet34(pretrained=pretrained)
    model = M2Net(base_model)

    return model


def res2dot5(num_classes=1, zero_init_residual=True, **kwargs):
    # model = res25.resnet34(pretrained=kwargs['pretrained'], num_classes=1, att=kwargs['att'],
    #                        model_path=kwargs['model_path'])
    model = res25.resnet34(num_classes=num_classes, zero_init_residual=zero_init_residual)
    if kwargs['pretrained']:
        import pathlib
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
        kwargs['init_weights'] = False
        path = kwargs['model_path']

        new_state_dict = OrderedDict()

        kwargs['init_weights'] = False
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder up to before the embedding layer
            if k.startswith('module.encoder') and not k.startswith('module.encoder.fc'):
                # remove prefix
                state_dict[k[len("module.encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict, strict=False)
        print("pretrained model loaded successfully! <============>")
        print("pretrained model name: %s" % kwargs['model_path'])

    return model


def swin2dot5(num_cls=1, pretrained=False, **kwargs):
    class swin_2dot5(nn.Module):
        def __init__(self, base_model, init_weights=True, num_classes=num_cls):
            super(swin_2dot5, self).__init__()
            self.siamese = False
            self.base_model = base_model
            # Remove the pooling layer and full connection layer #
            self.features_0 = self.base_model
            self.features_1 = self.base_model
            self.features_2 = self.base_model
            self.features_3 = self.base_model
            self.features_4 = self.base_model

            self.fc_0_1 = nn.Linear(384 * 5, 1)
            self.fc_1_1 = nn.Linear(384 * 5, 1)
            self.fc_2_1 = nn.Linear(384 * 5, 1)
            self.fc_3_1 = nn.Linear(384 * 5, 1)
            self.fc_4_1 = nn.Linear(384 * 6, 1)
            self.fc_0_2 = nn.Linear(384, 1)

            self.classifier = nn.Sequential(
                nn.Softmax(dim=2),
                nn.Linear(26, num_classes)
            )

            if init_weights:
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        timm.models.layers.trunc_normal_(m.weight, std=.02)
                        if isinstance(m, nn.Linear) and m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.LayerNorm):
                        nn.init.constant_(m.bias, 0)
                        nn.init.constant_(m.weight, 1.0)

        @autocast()
        def _forward_impl(self, x):
            temp_0, temp_1, temp_2, temp_3, temp_4 = ([] for i in range(5))
            import pdb
            pdb.set_trace()
            for i in range(0, 5):
                f = self.features_0(x[:, :, i, :, :])
                f = torch.flatten(f, start_dim=1)
                f = f[None, ...]
                temp_0.append(f)
            for i in range(5, 10):
                f = self.features_1(x[:, :, i, :, :])
                f = torch.flatten(f, start_dim=1)
                f = f[None, ...]
                temp_1.append(f)
            for i in range(10, 15):
                f = self.features_2(x[:, :, i, :, :])
                f = torch.flatten(f, start_dim=1)
                f = f[None, ...]
                temp_2.append(f)
            for i in range(15, 20):
                f = self.features_3(x[:, :, i, :, :])
                f = torch.flatten(f, start_dim=1)
                f = f[None, ...]
                temp_3.append(f)
            for i in range(20, 26):
                f = self.features_4(x[:, :, i, :, :])
                f = torch.flatten(f, start_dim=1)
                f = f[None, ...]
                temp_4.append(f)
            temp_0 = torch.squeeze(torch.stack(temp_0, 1), 0).permute(1, 0, 2)
            temp_1 = torch.squeeze(torch.stack(temp_1, 1), 0).permute(1, 0, 2)
            temp_2 = torch.squeeze(torch.stack(temp_2, 1), 0).permute(1, 0, 2)
            temp_3 = torch.squeeze(torch.stack(temp_3, 1), 0).permute(1, 0, 2)
            temp_4 = torch.squeeze(torch.stack(temp_4, 1), 0).permute(1, 0, 2)

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
                x_all = nn.functional.dropout(x_all, p=0.3)
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

    base_model = swin3d.SwinTransformer(img_size=[128, 128], patch_size=[2, 2], in_chans=1, embed_dim=48,
                                        depths=[2, 2, 18, 2], spatial_dims=2, num_classes=0,
                                        num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.0, use_checkpoint=True)
    model = swin_2dot5(base_model)
    pretrained_fc = False
    if pretrained:
        kwargs['init_weights'] = False
        # swin unetr weight: swin_unetr_model_swinvit.pt
        # swin Unet weight: swin_tiny_patch4_window7_224.pth
        model_path = kwargs['model_path'] + '/swin_tiny_patch4_window7_224.pth'
        if model_path is None:
            raise ValueError("please specify pretrained model path")
        checkpoint = torch.load(model_path)
        try:
            state_dict = checkpoint['state_dict']
        except KeyError:
            state_dict = checkpoint['model']
        model_dict = model.state_dict()
        if pretrained_fc:
            pretrained_dict = {k: v for k, v in state_dict.items() if
                               k in model_dict and not k.startswith("classifier")}
        else:
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("pretained weights provided and loaded")

    return model


