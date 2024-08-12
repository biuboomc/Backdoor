from torchvision import transforms
from torchvision.datasets import CIFAR10

import numpy as np
from torch.utils.data import DataLoader, Dataset
from base import CleanData
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms
import numpy as np
import os
BASE_DIR = Path(__file__).parent
import sys
import os
from PIL import Image
from tqdm import tqdm


def test(model, dataloader, device):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model.eval()
    acc = []
    loss = []
    batch_size = []
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            x, y = data
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss.append(criterion(out, y).item())
            batch_size.append(x.shape[0])
            acc.append((out.argmax(dim=1) == y).sum().item())
        acc = sum(acc) / sum(batch_size)
        # loss /= len(dataloader)
        loss = sum([loss[i] * batch_size[i] for i in range(len(loss))]) / sum(batch_size)
        return acc, loss


def l1_penalty(model):
    masks = model.get_masks()
    res = 0.
    for mask in masks:
        res += mask.abs().sum()
    return res


def clip_masks(model, threshold=1.0):
    masks = model.get_masks()
    with torch.no_grad():
        for mask in masks:
            mask.data.clamp_(0, threshold)


def test1(model, dataloader, device):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model.eval()
    acc = []
    loss = []
    batch_size = []
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            x, y = data
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss.append(criterion(out, y).item())
            batch_size.append(x.shape[0])
            acc.append((out.argmax(dim=1) == y).sum().item())
        acc = sum(acc) / sum(batch_size)
        # loss /= len(dataloader)
        loss = sum([loss[i] * batch_size[i] for i in range(len(loss))]) / sum(batch_size)
        return acc, loss

def train1(model, args, trainloader, test_loader = None, p_test_loader = None, mask = False, unlearn = False, mask_penalty = False):
    if mask:
        optimizer = torch.optim.SGD(model.get_masks(), lr=args.lr_mask)
    else:
        optimizer = torch.optim.SGD(model.get_params(), lr=args.lr1)
        mask_penalty = False
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    for i in range(1, args.epochs + 1):
        model.train()
        acc = 0.
        total_loss = 0.
        for data in tqdm(trainloader):
            x, y = data
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if unlearn:
                loss = -loss
            if mask_penalty:
                penalty = args.penalty * l1_penalty(model)
                # for debugging
                print(loss, penalty)
                loss += penalty
            loss.backward()
            optimizer.step()
            clip_masks(model)
            total_loss += loss.item()
            acc += (out.argmax(dim=1) == y).sum().item()
        acc /= len(trainloader.dataset)
        total_loss /= len(trainloader)
        args.logger.info("epoch:{}, acc:{}, total_loss:{}".format(i, acc, total_loss))

        model.eval()
        with torch.no_grad():
            if p_test_loader is not None:
                asr, total_loss = test(model, p_test_loader, args.device)
                args.logger.info("asr:{}, total_loss:{}".format(asr, total_loss))
            if test_loader is not None:
                ca, total_loss = test(model, test_loader, args.device)
                args.logger.info("ca:{}, total_loss:{}".format(ca, total_loss))


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]
        self.targets = [i[1] for i in self.images]

    def _create_class_idx_dict_train(self):
        classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt

class Subset(Dataset):
    def __init__(self, dataset, indices, transform = None):
        self.data = torch.utils.data.Subset(dataset, indices)
        self.targets = [dataset.targets[i] for i in indices]
        self.transform = transform
    def __getitem__(self, idx):
        image = self.data[idx][0]
        if self.transform is not None:
            image = self.transform(image)
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)

# def tinyimgnet_normalization():
#     return transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
# def cifar10_normalization():
#     return transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

def imagenet_normalization():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def load_cifar10(args, mode = 'defense'):
    transform_train = transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(10),
        transforms.ToTensor(),
        imagenet_normalization()
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224)),
        transforms.ToTensor(),
        imagenet_normalization()
    ])
    train_set = CIFAR10(BASE_DIR/'dataset', train=True, download=True)
    test_set = CIFAR10(BASE_DIR/'dataset', train=False)


    attack = CleanData

    indices = list(range(len(test_set)))
    np.random.shuffle(indices)
    benign_idx = indices[:args.defense_data]
    benign_test_set = Subset(test_set, benign_idx)

    defense_loader = DataLoader(dataset=CleanData(benign_test_set, mode= 'train', attack_target=args.target_label, transform = transform_test), batch_size=args.batch_size, num_workers=1, pin_memory=True, persistent_workers=True, shuffle=True)
    p_defense_loader = DataLoader(dataset=attack(benign_test_set, mode= 'test', attack_target=args.target_label, transform = transform_test), batch_size=args.batch_size, num_workers=1, pin_memory=True, persistent_workers=True, shuffle=True)
    return defense_loader, p_defense_loader