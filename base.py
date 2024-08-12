from torch.utils.data import Dataset
import torchvision
import torch
import numpy as np
class CleanData(Dataset):
    def __init__(self, dataset, mode = 'test', attack_target = 0, transform = None, pratio = 0.1):
        assert isinstance(dataset, Dataset)
        assert mode in ["train", "test"]
        if transform is None:
            self.transform = torchvision.transforms.Compose([])
        else:
            self.transform = transform
        self.dataset = dataset
        self.attack_target = attack_target
        self.mode = mode
        if mode == 'test':
            self._pop_original_class()
        if mode == 'train':
            length = len(self.dataset)
            p_num = int(length*pratio)
            tmp_list = list(range(length))
            np.random.shuffle(tmp_list)
            self.p_idx = tmp_list[:p_num]

    def _pop_original_class(self):
        indices = (
            (torch.tensor(self.dataset.targets)[..., None] != self.attack_target)
            .any(-1)
            .nonzero(as_tuple=True)[0]
        )
        from dataset import Subset
        subset = Subset(self.dataset, indices)
        self.dataset = subset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.transform(self.dataset[index][0]), self.dataset[index][1]
