import torch
import torch.nn as nn
from torchmeta.modules import MetaModule, MetaLinear
from global_utils import get_backbone

class MAMLNetwork(MetaModule):
    def __init__(self, backbone_name, input_channels, num_ways):
        super(MAMLNetwork, self).__init__()
        self.input_channels = input_channels
        self.num_ways = num_ways
                
        self.encoder = get_backbone(backbone_name) 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.classifier = MetaLinear(input_channels, num_ways)

    def forward(self, inputs, params=None, Emd=False):
        features = self.encoder(inputs, params=self.get_subdict(params, 'encoder'))
        features = self.avgpool(features) 
        features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        if Emd:
            return features, logits
        else:
            return logits

class RegularNetwork(nn.Module):
    def __init__(self, in_channels, hh, regular_type):
        super(RegularNetwork, self).__init__()
        
        if regular_type == 'MLP':
            self.fc1 = nn.Linear(in_channels, hh)
            self.fc2 = nn.Linear(hh, 1)

        elif regular_type == 'Flatten_FTF':
            self.fc1 = nn.Linear(in_channels ** 2, hh)
            self.fc2 = nn.Linear(hh, 1)

    def forward(self, features):
        features = torch.relu(self.fc1(features))
        features = nn.functional.softplus(self.fc2(features))
        return torch.mean(features)






