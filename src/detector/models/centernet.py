import torch.nn as nn
import torchvision

from .blocks import Up

_backbones = {
    "resnet18": (torchvision.models.resnet18, 512),
    "resnet34": (torchvision.models.resnet34, 512),
    "resnet50": (torchvision.models.resnet50, 2048),
    "resnet101": (torchvision.models.resnet101, 2048),
    "resnet152": (torchvision.models.resnet152, 2048),
    "mobilenet_v2": (torchvision.models.mobilenet_v2, 1280)
    # "mobilenet_v3_small": (torchvision.models.mobilenet_v3_small, 576),
    # "mobilenet_v3_large": (torchvision.models.mobilenet_v3_large, 960),
}


class CenterNet(nn.Module):
    def __init__(self, num_classes=1, backbone="resnet18", bilinear=True):
        super().__init__()
        # create backbone.
        basemodel = _backbones[backbone][0](pretrained=True)
        if backbone == "mobilenet_v2":
            layers = list(basemodel.children())[:-1]
        else:
            layers = list(basemodel.children())[:-2]
        basemodel = nn.Sequential(*layers)
        # set basemodel
        self.base_model = basemodel
        self.bilinear = bilinear

        num_ch = _backbones[backbone][1]

        self.up1 = Up(num_ch, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 256, bilinear)
        # output classification
        self.out_classification = nn.Conv2d(256, num_classes, 1)
        # output residue
        self.out_residue = nn.Conv2d(256, 2, 1)

    def forward(self, x):
        x = self.base_model(x)
        # Add positional info
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        c = self.out_classification(x)
        r = self.out_residue(x)
        return c, r
