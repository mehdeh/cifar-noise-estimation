"""ResNet implementation for noise level estimation.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18 and ResNet-34."""
    
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        """
        Initialize BasicBlock.
        
        Args:
            in_planes (int): Number of input channels
            planes (int): Number of output channels
            stride (int): Stride for the first convolution
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        """Forward pass through the basic block."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck residual block for ResNet-50/101/152."""
    
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        """
        Initialize Bottleneck block.
        
        Args:
            in_planes (int): Number of input channels
            planes (int): Number of output channels
            stride (int): Stride for the convolution
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        """Forward pass through the bottleneck block."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet architecture for noise level estimation."""
    
    def __init__(self, block, num_blocks, num_classes=1):
        """
        Initialize ResNet.
        
        Args:
            block: Block type (BasicBlock or Bottleneck)
            num_blocks (list): Number of blocks in each layer
            num_classes (int): Number of output classes (1 for regression)
        """
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """Create a layer with multiple residual blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through ResNet.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, 32, 32)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, num_classes)
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=1):
    """
    Create ResNet-18 model.
    
    Args:
        num_classes (int): Number of output classes
        
    Returns:
        ResNet: ResNet-18 model
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes=1):
    """
    Create ResNet-34 model.
    
    Args:
        num_classes (int): Number of output classes
        
    Returns:
        ResNet: ResNet-34 model
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes=1):
    """
    Create ResNet-50 model.
    
    Args:
        num_classes (int): Number of output classes
        
    Returns:
        ResNet: ResNet-50 model
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes=1):
    """
    Create ResNet-101 model.
    
    Args:
        num_classes (int): Number of output classes
        
    Returns:
        ResNet: ResNet-101 model
    """
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes=1):
    """
    Create ResNet-152 model.
    
    Args:
        num_classes (int): Number of output classes
        
    Returns:
        ResNet: ResNet-152 model
    """
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)

