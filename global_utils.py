import numpy as np
import scipy.stats
from torchvision import transforms

def get_backbone(name, state_dict=None):
    if name == 'conv4':
        from backbones import conv4
        backbone = conv4()
    elif name == 'resnet10':
        from backbones import resnet10
        backbone = resnet10()
    elif name == 'metaconv4':
        from backbones import metaconv4
        backbone = metaconv4()
    elif name == 'metaresnet10':
        from backbones import metaresnet10
        backbone = metaresnet10()
    else:
        raise ValueError('Non-supported Backbone.')
    if state_dict is not None:
        backbone.load_state_dict(state_dict)
    return backbone
    
class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0
    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1
    def item(self):
        return self.v

class Mean_confidence_interval():
    def __init__(self, confidence=0.95):
        self.list = []
        self.confidence = confidence
        self.n = 0
    def add(self, x):
        self.list.append(x)
        self.n += 1
    def item(self, return_str=False):
        mean, standard_error = np.mean(self.list), scipy.stats.sem(self.list)
        h = standard_error * scipy.stats.t._ppf((1 + self.confidence) / 2, self.n - 1)
        if return_str:
            return '{0:.2f}; {1:.2f}'.format(mean * 100, h * 100)
        else:
            return mean

def get_outputs_c(backbone):
    c_dict = {
        'conv4': 64,
        'resnet10': 512,
        'metaconv4': 64,
        'metaresnet10': 512
    }
    c = c_dict[backbone]
    
    return c

def get_dataset(args, dataset_name, phase):
    if dataset_name == 'cub':
        from torchmeta.datasets.helpers import cub as dataset_helper
        image_size = 84
        padding_len = 8
    elif dataset_name == 'miniimagenet':
        from torchmeta.datasets.helpers import miniimagenet as dataset_helper
        image_size = 84
        padding_len = 8
    elif dataset_name == 'tieredimagenet':
        from torchmeta.datasets.helpers import tieredimagenet as dataset_helper
        image_size = 84
        padding_len = 8
    elif dataset_name == 'cifar_fs':
        from torchmeta.datasets.helpers import cifar_fs as dataset_helper
        image_size = 84
        padding_len = 8
    else:
        raise ValueError('Non-supported Dataset.')

    if dataset_name == 'tieredimagenet':
        if args.augment and phase == 'train':
            transforms_list = [
                transforms.RandomCrop(image_size, padding=padding_len),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]
        else:
            transforms_list = [
                transforms.ToTensor()
            ]
    else:
        if args.augment and phase == 'train':
            transforms_list = [
                transforms.RandomResizedCrop(image_size),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]
        else:
            transforms_list = [
                transforms.Resize(image_size+padding_len),
                transforms.CenterCrop(image_size),
                transforms.ToTensor()
            ]

    transforms_list = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])

    # get datasets
    dataset = dataset_helper(args.data_folder,
                            shots=args.num_shots,
                            ways=args.num_ways,
                            shuffle=(phase == 'train'),
                            test_shots=args.test_shots,
                            meta_split=phase,
                            download=args.download,
                            transform=transforms_list)

    return dataset

