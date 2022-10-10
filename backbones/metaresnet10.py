import torch.nn as nn
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                        MetaBatchNorm2d)

def conv3x3(in_planes, out_planes, stride=1):
    return MetaConv2d(in_planes, out_planes, kernel_size=3, stride=stride, 
                    padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return MetaConv2d(in_planes, out_planes, kernel_size=1, stride=stride, 
                    bias=False)

class BasicBlock(MetaModule):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = MetaBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = MetaBatchNorm2d(planes)
        self.downsample = downsample    
        self.stride = stride

    def forward(self, inputs, params=None):
        identity = inputs

        outputs = self.conv1(inputs, params=self.get_subdict(params, 'conv1'))
        outputs = self.bn1(outputs, params=self.get_subdict(params, 'bn1'))
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs, params=self.get_subdict(params, 'conv2'))
        outputs = self.bn2(outputs, params=self.get_subdict(params, 'bn2'))

        if self.downsample is not None:
            identity = self.downsample(inputs)

        outputs += identity
        outputs = self.relu(outputs)

        return outputs

class ResNet(MetaModule):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = MetaConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = MetaBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self._init_conv()

        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = MetaSequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                MetaBatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))

        return MetaSequential(*layers)

    def _init_conv(self):
        for m in self.modules():
            if isinstance(m, MetaConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, MetaBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, inputs, params=None):
        outputs = self.conv1(inputs,params=self.get_subdict(params, 'conv1'))
        outputs = self.bn1(outputs, params=self.get_subdict(params, 'bn1'))
        outputs = self.relu(outputs)

        outputs = self.layer1(outputs, params=self.get_subdict(params, 'layer1'))
        outputs = self.layer2(outputs, params=self.get_subdict(params, 'layer2'))
        outputs = self.layer3(outputs, params=self.get_subdict(params, 'layer3'))
        outputs = self.layer4(outputs, params=self.get_subdict(params, 'layer4'))

        return outputs


def metaresnet10(**kwargs):
    return ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)


    