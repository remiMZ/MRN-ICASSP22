import torch.nn as nn
from torchmeta.modules import MetaModule, MetaLinear
import sys
sys.path.append("..")
from global_utils import get_backbone

class MAMLNetwork(MetaModule):
    def __init__(self, backbone_name, input_channels, num_ways):
        super(MAMLNetwork, self).__init__()
        self.input_channels = input_channels
        self.num_ways = num_ways
                
        self.encoder = get_backbone(backbone_name) 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.classifier = MetaLinear(input_channels, num_ways)

    def forward(self, inputs, params=None):
        features = self.encoder(inputs, params=self.get_subdict(params, 'encoder'))
        features = self.avg_pool(features)
        features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits
