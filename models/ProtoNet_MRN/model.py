import torch
import torch.nn as nn
import sys 
sys.path.append("..") 
sys.path.append("../..")

from global_utils import get_backbone

class PrototypicalNetwork(nn.Module):
    def __init__(self, backbone_name):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = get_backbone(backbone_name)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, inputs):
        embeddings = self.encoder(inputs.view(-1, *inputs.shape[2:]))
        embeddings = self.avgpool(embeddings)
        return embeddings.view(*inputs.shape[:2], -1)

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
