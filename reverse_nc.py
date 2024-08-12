import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.nn import Parameter
import matplotlib.pyplot as plt
import logging
from args import init_and_load_args
from dataset import load_cifar10
from torchvision import transforms
import torch
from torch.utils.data.dataloader import DataLoader
from vit import TiViT
import numpy as np
from PIL import Image

logging.getLogger('matplotlib').setLevel(logging.WARNING)

progress = 0.00
def update_progress(epoches):
    global progress
    progress += 10/epoches

def mask2attn(matrix, args):
    patch_sums = torch.zeros((14, 14))
    # print(matrix.shape)

    patch_size = 16

    for i in range(0, 224, patch_size):
        for j in range(0, 224, patch_size):
            patch = matrix[i:i+patch_size, j:j+patch_size]
            patch_sums[i//patch_size, j//patch_size] = torch.sum(patch).item() / (patch_size * patch_size)
    
    vector_196 = patch_sums.flatten()
    matrix_196 = vector_196.reshape(196, 1) @ vector_196.reshape(1, 196)
    matrix_197 = np.pad(matrix_196, ((1, 0), (1, 0)), 'constant', constant_values=0)

    matrix_197 = torch.from_numpy(matrix_197)
    matrix_197.requires_grad_(True)

    return matrix_197.to(args.device)

def train_trigger(model, target_label, train_loader, normalizer, args):
    w, h = [224, 224]
    # print(w,h)
    trigger = torch.ones((w, h), requires_grad=True, device=args.device)
    mask = torch.ones((w, h), device=args.device)/2
    mask.requires_grad_(True)

    w,h = 197, 197
    mask_attn = torch.zeros((w, h), device=args.device)
    # torch.clamp_(mask_attn, 0, 0.1)
    mask_attn.requires_grad_(True)

    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam([trigger, mask], lr=0.01)
    optimizer = torch.optim.Adam([mask_attn, trigger, mask], lr=0.01)
    model.eval()

    epochs = 2
    lam = 0.01

    pbar = tqdm(total=epochs, desc="Training Progress")
    for i in range(epochs):
        torch.cuda.empty_cache()
        asr = 0.
        for img, label in train_loader:
            optimizer.zero_grad()
            img = img.to(args.device)
            t_img = (1 - mask) * img + mask * trigger
            t_img = normalizer(t_img)
            y_pred = model(t_img, [1, (mask_attn, mask2attn(mask, args))])
            y_target = torch.full_like(label, target_label).to(args.device)
            loss = criterion(y_pred, y_target) + lam * torch.norm(mask, 1)
            asr += (y_pred.argmax(1) == y_target).sum().item()

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                torch.clamp_(mask, 0, 1)
                torch.clamp_(mask_attn, 0, 0.2)
                torch.clamp_(trigger, 0, 1)
                norm1 = torch.sum(torch.abs(mask)).item()

        asr /= len(train_loader.dataset)
        pbar.set_description(f"Loss: {loss.item():.4f}, Norm1: {norm1:.4f}, ASR: {asr:.4f}")
        pbar.update(1)

        update_progress(epochs)
    pbar.close()

    return trigger.detach().cpu(), mask.detach().cpu(), mask_attn.detach().cpu(), mask2attn(mask, args).detach().cpu(), norm1, asr

def reverse_trigger(model, dataloader, normalizer, args, folder_path, labels = None):
    triggers = []
    masks = []
    norms1 = []
    norms2 = []
    asrs = []
    if labels is None:
        labels = range(args.num_classes)

    for label in labels:
        trigger, mask, mask_attn, mask_attn_T, norm1, acc = train_trigger(model, label, dataloader, normalizer, args)
        trigger = trigger.numpy()
        # plt.axis("off")
        # plt.imshow(trigger)
        im = Image.fromarray(trigger)
        if im.mode == 'F':
            im = (im - np.min(im)) / (np.max(im) - np.min(im)) * 255  # 归一化到0-255
            im = im.astype(np.uint8)  # 转换为整数类型
            # 转换为灰度图
        im = Image.fromarray(im).convert('L')
        im.save(folder_path+'/trigger_{}.png'.format(label))

        # mask = mask.squeeze(0).numpy()
        mask = mask.numpy()
        # plt.axis("off")
        # plt.imshow(mask)
        im = Image.fromarray(mask)
        if im.mode == 'F':
            im = (im - np.min(im)) / (np.max(im) - np.min(im)) * 255  # 归一化到0-255
            im = im.astype(np.uint8)  # 转换为整数类型
            # 转换为灰度图
        im = Image.fromarray(im).convert('L')
        im.save(folder_path+'/mask_{}.png'.format(label))

        mask_attn = mask_attn.numpy()
        # plt.axis("off")
        # plt.imshow(mask_attn)
        im = Image.fromarray(mask_attn)
        if im.mode == 'F':
            im = (im - np.min(im)) / (np.max(im) - np.min(im)) * 255  # 归一化到0-255
            im = im.astype(np.uint8)  # 转换为整数类型
            # 转换为灰度图
        im = Image.fromarray(im).convert('L')
        im.save(folder_path+'/perturb_attn_{}.png'.format(label))


        mask_attn_T = mask_attn_T.numpy()
        # plt.axis("off")
        # plt.imshow(mask_attn_T)
        im = Image.fromarray(mask_attn_T)
        if im.mode == 'F':
            im = (im - np.min(im)) / (np.max(im) - np.min(im)) * 255  # 归一化到0-255
            im = im.astype(np.uint8)  # 转换为整数类型
            # 转换为灰度图
        im = Image.fromarray(im).convert('L')
        im.save(folder_path+'/mask_attn_{}.png'.format(label))

        norms1.append(norm1)
        asrs.append(acc)
        triggers.append(trigger)
        masks.append(mask)

    return triggers, masks, norms1, asrs

if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = init_and_load_args('nc.yaml')
    model = TiViT(args.num_classes, masked=True).to(args.device)
    state_dict = torch.load('bd.pth')
    model.load_state_dict(state_dict, strict=False)

    testset, ptestset = load_cifar10(args)
    testset.dataset.transform = transforms.Compose([
        transforms.Resize((224)),
        transforms.ToTensor(),
    ])
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    folder_path = 'result'
    import os
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    triggers, masks, norms1, asrs = reverse_trigger(model, testset, normalizer, args, folder_path)
    np.save("norm1_bad_attn.npy", norms1)
    np.save("asrs_bad_attn.npy", asrs)

