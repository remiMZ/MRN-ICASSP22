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

