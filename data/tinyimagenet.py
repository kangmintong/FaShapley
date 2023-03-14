import os
import numpy as np

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import multiprocessing

# NOTE: Each dataset class must have public norm_layer, tr_train, tr_test objects.
# These are needed for ood/semi-supervised dataset used alongwith in the training and eval.
class tinyimagenet:

    def __init__(self, args, normalize=True):
        self.args = args
        self.norm_layer = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])

        self.tr_train = [
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomCrop(56, padding_mode='edge'),
                      transforms.ToTensor(),
                                             ]
        self.tr_test = [
                         # transforms.RandomResizedCrop(64, scale=(0.875, 0.875), ratio=(1., 1.)),
                         transforms.CenterCrop(56),
                         transforms.ToTensor()
                         ]

        if normalize:
            self.tr_train.append(self.norm_layer)
            self.tr_test.append(self.norm_layer)

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

    def data_loaders(self, **kwargs):
        train_data = datasets.ImageFolder('./data/tinyImageNet/tiny-imagenet-200' + '/train',
                                          transform=self.tr_train)
        test_data = datasets.ImageFolder(
            './data/tinyImageNet/tiny-imagenet-200' + '/val',
            transform=self.tr_test)

        train_data = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True, pin_memory=True,
                                                 num_workers=min(multiprocessing.cpu_count(), 4))
        test_data = torch.utils.data.DataLoader(test_data, batch_size=128 // 5, pin_memory=True,
                                                num_workers=min(multiprocessing.cpu_count(), 4))

        return train_data, test_data


    def shapley_loaders(self, **kwargs):
        train_data = datasets.ImageFolder('./data/tinyImageNet/tiny-imagenet-200' + '/train',
                                          transform=self.tr_train)

        print(f'len(train_data): {len(train_data)}')
        data_fraction = 1000 / len(train_data)
        subset_indices = np.random.permutation(np.arange(len(train_data)))[
                         : int(data_fraction * len(train_data))
                         ]
        train_data = torch.utils.data.DataLoader(train_data, batch_size=1000, shuffle=True, pin_memory=True,
                                                 num_workers=min(multiprocessing.cpu_count(), 4),sampler=SubsetRandomSampler(subset_indices),)



        return train_data