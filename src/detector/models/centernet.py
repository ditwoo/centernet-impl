import torch.nn as nn
import torchvision

from .blocks import Up


class ResNetCenterNet(nn.Module):
    def __init__(self, num_classes=1, model_name="resnet18"):
        super().__init__()
        # create backbone.
        basemodel = torchvision.models.resnet18(pretrained=True)  # turn this on for training
        basemodel = nn.Sequential(*list(basemodel.children())[:-2])
        # set basemodel
        self.base_model = basemodel

        if model_name == "resnet34" or model_name == "resnet18":
            num_ch = 512
        else:
            num_ch = 2048

        self.up1 = Up(num_ch, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 256)
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
