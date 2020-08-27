# pytorch and numpy libs
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# import the grouping unit
import sys
import os
sys.path.append(os.path.abspath('../common'))
from grouping import GroupingUnit

# wrap up the convolution
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# Bottleneck of standard ResNet50/101
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# Basicneck of standard ResNet18/34
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# Bottleneck of standard ResNet50/101, with kernel size equal to 1
class Bottleneck1x1(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck1x1, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, num_parts=32):
        super(ResNet, self).__init__()

        # model params
        self.inplanes = 64
        self.n_parts = num_parts
        self.num_classes = num_classes

        # modules in original resnet as the feature extractor
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        # the grouping module
        self.grouping = GroupingUnit(256*block.expansion, num_parts)
        self.grouping.reset_parameters(init_weight=None, init_smooth_factor=None)

        # post-processing bottleneck block for the region features
        self.post_block = nn.Sequential(
            Bottleneck1x1(1024, 512, stride=1, downsample = nn.Sequential(
                nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(2048))),
            Bottleneck1x1(2048, 512, stride=1),
            Bottleneck1x1(2048, 512, stride=1),
            Bottleneck1x1(2048, 512, stride=1),
        )


        # an attention for each classification head
        self.attconv = nn.Sequential(
            Bottleneck1x1(1024, 256, stride=1),
            Bottleneck1x1(1024, 256, stride=1),
            nn.Conv2d(1024, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

        # the final batchnorm
        self.groupingbn = nn.BatchNorm2d(2048)

        # linear classifier for each attribute
        self.mylinear = nn.Linear(2048, num_classes)

        # initialize convolutional layers with kaiming_normal_, BatchNorm with weight 1, bias 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize the last bn in residual blocks with weight zero
        for m in self.modules():
            if isinstance(m, Bottleneck) or isinstance(m, Bottleneck1x1):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    # layer generation for resnet backbone
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        # the resnet backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # grouping module upon the feature maps outputed by the backbone
        region_feature, assign = self.grouping(x)
        region_feature = region_feature.contiguous().unsqueeze(3)

        # generate attention
        att = self.attconv(region_feature)
        att = F.softmax(att, dim=2)

        # non-linear layers over the region features
        region_feature = self.post_block(region_feature)

        # attention-based classification
        # apply the attention on the features
        out = region_feature * att
        out = out.contiguous().squeeze(3)

        # average all region features into one vector based on the attention
        out = F.avg_pool1d(out, self.n_parts) * self.n_parts
        out = out.contiguous().unsqueeze(3)

        # final bn
        out = self.groupingbn(out)

        # linear classifier
        out = out.contiguous().view(out.size(0), -1)
        out = self.mylinear(out)

        return out, att, assign

# model wrapper
def ResNet50(num_classes, num_parts=5):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, num_parts)

def ResNet101(num_classes, num_parts=5):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, num_parts)
