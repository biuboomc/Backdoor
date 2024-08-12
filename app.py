from flask import Flask, request, jsonify, render_template, url_for, send_from_directory
import torch
from torchvision import transforms
from PIL import Image
import io
import os
from vit import TiViT
import argparse
import matplotlib.pyplot as plt
import logging
from args import init_and_load_args
from dataset import load_cifar10
from torchvision import transforms
import os
from enum import Enum
import numpy as np
from PIL import Image
from tqdm import tqdm
from tqdm import tqdm

import yaml

from dataset import load_cifar10, test1, train1
import torch
import optimize_mask_cifar



os.environ['HF_ENDPOINT']='https://hf-mirror.com'
logging.getLogger('matplotlib').setLevel(logging.WARNING)

app = Flask(__name__)
progress = 0
running = 0
shut = 0

# 为upload和anp设置不同的上传路径
app.config['UPLOAD_FOLDER_UPLOAD'] = 'uploads/'
app.config['UPLOAD_FOLDER_ANP'] = 'anp/'
app.config['UPLOAD_FOLDER_RNP'] = 'RNP/'

ALLOWED_EXTENSIONS = {'h5', 'pth', 'pt', 'csv'}




def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class Status(Enum):
    NO_BD = 1  # 无后门
    BD = 2  # 后门模型
    Salient_BD_Risk = 3  # 显著后门风险
    BD_Risk = 4  # 后门风险


def is_backdoor(norms, asrs, masks):
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)

    high_asr_count = sum(asr > 0.9 for asr in asrs)
    if high_asr_count == 0:
        return Status.NO_BD, 0

    for norm, asr in zip(norms, asrs):
        if asr > 0.95 and norm < mean_norm - std_norm:
            idx = list(norms).index(norm)
            return Status.BD, idx
        elif 0.8 < asr <= 0.95 and norm < mean_norm - std_norm:
            idx = list(norms).index(norm)
            state, info = if_BD_risk(masks[idx], idx)
            return state, info

    if all(asr > 0.8 for asr in asrs):
        return Status.BD_Risk, 0
    else:
        return Status.NO_BD, 0

    return Status.NO_BD, 0

def normalize_and_threshold(vector, th_high, th_low):
    # 归一化
    min_val = np.min(vector)
    max_val = np.max(vector)
    normalized_vector = (vector - min_val) / (max_val - min_val)

    # 阈值处理
    thresholded_vector = np.where(normalized_vector > th_high, 1, 0)
    thresholded_vector = np.where(normalized_vector < th_low, 1, thresholded_vector)

    return normalized_vector, thresholded_vector


def if_BD_risk(mask, idx, patch_size=16, th_high=0.8, th_low=0.2):
    mask = Image.fromarray(mask)
    mask = mask.resize((224, 224))
    mask = np.array(mask)

    # 计算patches的累加值
    S = np.zeros((14, 14))  # 224/16 = 14
    for i in range(0, 224, patch_size):
        for j in range(0, 224, patch_size):
            patch = mask[i:i + patch_size, j:j + patch_size]
            S[i // patch_size, j // patch_size] = np.sum(patch)

    # 将S展平
    vector = S.flatten()
    if np.std(vector) == 0:
        return Status.BD_Risk, 0
    # 归一化和阈值处理
    normalized_vector, thresholded_vector = normalize_and_threshold(vector, th_high, th_low)
    # print(thresholded_vector)
    # 计算范数差异
    diff_norm = np.linalg.norm(thresholded_vector, 1)
    normalized = np.linalg.norm(normalized_vector, 1)
    print(diff_norm, normalized, np.std(normalized_vector*100))
    # 检查是否匹配

    if len(vector) - diff_norm <= 1:
        return Status.BD, idx

    return Status.BD_Risk, 0


def save_mask_scores(model, filename="mask_scores.pth"):
    args = init_and_load_args('nc.yaml')
    mask_dict = dict()
    for key, value in model.state_dict().items():
        if 'mask' in key:
            mask_dict[key] = value
    print(args.log_dir)
    print(filename)
    print(args.log_dir/filename)
    # if "OneDrive" in str(args.log_dir/filename):
    #     import re
    #     path = re.sub(r"\\OneDrive[^\\]*\\", r"\\", str(args.log_dir/filename))

    torch.save(mask_dict, args.log_dir/filename)
    return mask_dict

def update_progress(epoches):
    global progress
    if progress<=100:
        progress += 10 / epoches


def mask2attn(matrix, args):
    patch_sums = torch.zeros((14, 14))
    # print(matrix.shape)

    patch_size = 16

    for i in range(0, 224, patch_size):
        for j in range(0, 224, patch_size):
            patch = matrix[i:i + patch_size, j:j + patch_size]
            patch_sums[i // patch_size, j // patch_size] = torch.sum(patch).item() / (patch_size * patch_size)

    vector_196 = patch_sums.flatten()
    matrix_196 = vector_196.reshape(196, 1) @ vector_196.reshape(1, 196)
    matrix_197 = np.pad(matrix_196, ((1, 0), (1, 0)), 'constant', constant_values=0)

    matrix_197 = torch.from_numpy(matrix_197)
    matrix_197.requires_grad_(True)

    return matrix_197.to(args.device)


def train_trigger(model, target_label, train_loader, normalizer, args, idx):
    w, h = [224, 224]
    trigger = torch.ones((w, h), requires_grad=True, device=args.device)
    mask = torch.ones((w, h), device=args.device) / 2
    mask.requires_grad_(True)

    w, h = 197, 197
    mask_attn = torch.zeros((w, h), device=args.device)
    mask_attn.requires_grad_(True)

    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam([trigger, mask], lr=0.01)
    optimizer = torch.optim.Adam([mask, mask_attn, trigger], lr=0.01)
    model.eval()

    epochs = 50   #200
    lam = 0.02  #0.01

    pbar = tqdm(total=epochs, desc="Training Progress")
    for i in range(epochs):
        torch.cuda.empty_cache()
        asr = 0.
        for img, label in train_loader:
            if check(idx):
                return True, None, None, None, None, None, None
            optimizer.zero_grad()
            img = img.to(args.device)
            # print(img.shape)
            t_img = (1 - mask) * img + mask * trigger
            t_img = normalizer(t_img)
            y_pred = model(t_img, [1, (mask_attn, mask2attn(mask, args))])
            y_target = torch.full_like(label, target_label).to(args.device)
            # print(y_pred)
            loss = criterion(y_pred, y_target) + lam * torch.norm(mask, 1)
            asr += (y_pred.argmax(1) == y_target).sum().item()

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                torch.clamp_(mask, 0, 1)
                torch.clamp_(mask_attn, 0, 0.1)
                torch.clamp_(trigger, 0, 1)
                norm1 = torch.sum(torch.abs(mask)).item()

        asr /= len(train_loader.dataset)
        pbar.set_description(f"Label:{target_label}, Loss: {loss.item():.4f}, Norm1: {norm1:.4f}, ASR: {asr:.4f}")
        pbar.update(1)

        update_progress(epochs)
    pbar.close()

    return False, trigger.detach().cpu(), mask.detach().cpu(), mask_attn.detach().cpu(), mask2attn(mask,
                                                                                            args).detach().cpu(), norm1, asr


def reverse_trigger(model, dataloader, normalizer, args, folder_path, idx, labels=None):
    global progress
    progress = 0
    triggers = []
    masks = []
    norms1 = []
    asrs = []
    if labels is None:
        labels = range(args.num_classes)

    for label in labels:
        check_flag, trigger, mask, mask_attn, mask_attn_T, norm1, acc = train_trigger(model, label, dataloader, normalizer, args, idx)
        if check_flag:
            return check_flag, None, None, None, None
        trigger = trigger.numpy()
        # plt.axis("off")
        # plt.imshow(trigger)
        im = Image.fromarray(trigger)
        if im.mode == 'F':
            im = (im - np.min(im)) / (np.max(im) - np.min(im)) * 255  # 归一化到0-255
            im = im.astype(np.uint8)  # 转换为整数类型
            # 转换为灰度图
        im = Image.fromarray(im).convert('L')
        im.save(folder_path + '/trigger_{}.png'.format(label))

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
        im.save(folder_path + '/mask_{}.png'.format(label))

        mask_attn = mask_attn.numpy()
        # plt.axis("off")
        # plt.imshow(mask_attn)
        im = Image.fromarray(mask_attn)
        if im.mode == 'F':
            im = (im - np.min(im)) / (np.max(im) - np.min(im)) * 255  # 归一化到0-255
            im = im.astype(np.uint8)  # 转换为整数类型
            # 转换为灰度图
        im = Image.fromarray(im).convert('L')
        im.save(folder_path + '/perturb_attn_{}.png'.format(label))

        mask_attn_T = mask_attn_T.numpy()
        # plt.axis("off")
        # plt.imshow(mask_attn_T)
        im = Image.fromarray(mask_attn_T)
        if im.mode == 'F':
            im = (im - np.min(im)) / (np.max(im) - np.min(im)) * 255  # 归一化到0-255
            im = im.astype(np.uint8)  # 转换为整数类型
            # 转换为灰度图
        im = Image.fromarray(im).convert('L')
        im.save(folder_path + '/mask_attn_{}.png'.format(label))

        norms1.append(norm1)
        asrs.append(acc)
        triggers.append(trigger)
        masks.append(mask)

    return False, triggers, masks, norms1, asrs

def prune_with_scores(model, mask_scores, testloader, ptestloader, device, interval=20):
    args = init_and_load_args('nc.yaml')
    if isinstance(mask_scores, str):
        mask_scores = torch.load(args.log_dir / mask_scores)
    # print(mask_scores)
    scores = []
    for key, value in mask_scores.items():
        value = value.squeeze()
        for i in range(len(value)):
            scores.append((key, i, value[i]))
    scores = sorted(scores, key=lambda x: x[2])
    # print(scores)
    for i, neuron in enumerate(scores):
        key, idx, score = neuron
        state_dict = model.state_dict()
        state_dict[key][0, 0, 0, 0, idx] = 0.
        model.load_state_dict(state_dict, strict=False)
        if i % interval == 0:
            ca, loss = test1(model, testloader, device)
            asr, ploss = test1(model, ptestloader, device)
            # ca 先不放
            print("{}, idx: {}, mask: {:.4f}, loss: {:.4f}, asr: {:.4f}, ploss: {:.4f}".format(key, idx,
                                                                                                           score,
                                                                                                           loss, asr,
                                                                                                           ploss))
    pruned_model_path = os.path.join(args.log_dir, 'pruned_model.pth')
    torch.save(model.state_dict(), pruned_model_path)
    return pruned_model_path


@app.route('/get_progress', methods=['GET'])
def get_progress():
    # print(progress)
    return jsonify({'percent': int(progress)})


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/introduction')
def introduction():
    return render_template('templates/introduction.html')

@app.route('/publications')
def publications():
    return render_template('publications.html')

@app.route('/cv')
def cv():
    return render_template('cv.html')

def check(idx):
    global running
    if (idx < running):
        return True
    else:
        return False

@app.route('/upload', methods=['POST'])
def upload_file():
    global progress
    global running

    idx = running+1
    running += 1
    progress = 0
    print("Received POST request at /upload")
    if 'modelFile' not in request.files:
        return jsonify({'error': 'No model file part'}), 400
    model_file = request.files['modelFile']
    print(model_file)
    if model_file.filename == '':
        return jsonify({'error': 'No selected model file'}), 400
    if model_file and allowed_file(model_file.filename):
        model_path = os.path.join(app.config['UPLOAD_FOLDER_UPLOAD'], model_file.filename)
        model_file.save(model_path)


        torch.cuda.empty_cache()
        args = init_and_load_args('nc.yaml')
        logger = args.logger
        logger.info('----------- Load Model {} to Device {} --------------'.format(args.model, args.device))
        logger.info('----------- Load Dataset {} with Attack {} --------------'.format(args.dataset, args.attack))
        model = TiViT(args.num_classes, masked=True).to(args.device)
        state_dict = torch.load(model_path, map_location=args.device)
        model.load_state_dict(state_dict, strict=False)

        testset, ptestset = load_cifar10(args, mode='defense')

        # prune
        defense_loader, pdefense_loader = load_cifar10(args, mode='defense')


        testset.dataset.transform = transforms.Compose([
            transforms.Resize((224)),
            transforms.ToTensor(),
        ])

        normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        folder_path = 'static/result'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        check_flag, triggers, masks, norms, asrs = reverse_trigger(model, testset, normalizer, args, folder_path, idx)
        if check_flag:
            return jsonify({'error': 'abort', 'idx': idx}), 400

        state = is_backdoor(norms, asrs, masks)
        if_bd, label = state
        if if_bd == Status.NO_BD:
            words = "判断无后门"
        elif if_bd == Status.BD:
            words = f"判断有后门，后门标签为{label}"
        else:
            words = f"判断存在后门风险，需要小心使用"

        mask_url = url_for('static', filename=f'result/mask_{label}.png')
        mean_asr = np.mean(asrs) * 100
        mean = np.mean(norms)
        std = np.std(norms)
        min = np.min(norms)
        if min < mean - 2*std:
            compare = "< mean - 2*std"
        elif min < mean - std:
            compare = "< mean - std"
        else:
            compare = ">= mean - std"

        print(state)

        return jsonify({"mean_asr": mean_asr, "mean": mean, "std": std, "min": min, "words": words, "compare": compare, "mask": mask_url})

    return jsonify({'error': 'Invalid model file format'}), 400

# @app.route('/anp', methods=['POST'])
# def anp():
#     global progress
#     global running
#     idx = running + 1
#     running += 1
#     progress = 0
#     print("Received POST request at /anp")
#
#     if 'RNPModelFile' not in request.files:
#         return jsonify({'error': 'No model file part'}), 400
#     model_file = request.files['RNPModelFile']
#     print(model_file)
#     if model_file.filename == '':
#         return jsonify({'error': 'No selected model file'}), 400
#     if model_file and allowed_file(model_file.filename):
#         model_path = os.path.join(app.config['UPLOAD_FOLDER_ANP'], model_file.filename)
#         model_file.save(model_path)
#
#         # 加载模型
#         args = init_and_load_args('nc.yaml')
#         logger = args.logger
#         logger.info('----------- Load Model {} to Device {} --------------'.format(args.model, args.device))
#         model = TiViT(args.num_classes, masked=True).to(args.device)
#         state_dict = torch.load(model_path, map_location=args.device)
#         model.load_state_dict(state_dict, strict=False)
#
#         defense_loader, pdefense_loader = load_cifar10(args, mode='defense')
#         testset, ptestset = load_cifar10(args, mode='test')
#         train1(model, args, defense_loader, testset, ptestset, unlearn=True, mask=False)
#         train1(model, args, defense_loader, testset, ptestset, unlearn=False, mask=True, mask_penalty=False)
#         save_mask_scores(model, "mask_scores.pth")
#         # 假设mask_scores.pth文件在args.log_dir目录下
#         mask_scores_path = args.log_dir / "mask_scores.pth"
#
#         pruned_model_path = prune_with_scores(model, "mask_scores.pth", testset, ptestset, args.device)
#
#         # 返回处理结果
#         return jsonify({
#             'progress': progress,
#             'pruned_model_path': pruned_model_path
#         })
#
#     return jsonify({'error': 'Invalid model file format'}), 400

@app.route('/RNP', methods=['POST'])
def RNP():
    global progress
    global running
    idx = running + 1
    running += 1
    progress = 0
    print("Received POST request at /RNP")

    if 'RNPModelFile' not in request.files:
        return jsonify({'error': 'No model file part'}), 400
    model_file = request.files['RNPModelFile']
    print(model_file)
    if model_file.filename == '':
        return jsonify({'error': 'No selected model file'}), 400
    if model_file and allowed_file(model_file.filename):
        model_path = os.path.join(app.config['UPLOAD_FOLDER_RNP'], model_file.filename)
        model_file.save(model_path)

        # 加载模型
        args = init_and_load_args('nc.yaml')
        logger = args.logger
        logger.info('----------- Load Model {} to Device {} --------------'.format(args.model, args.device))
        model = TiViT(args.num_classes, masked=True).to(args.device)
        state_dict = torch.load(model_path, map_location=args.device)
        model.load_state_dict(state_dict, strict=False)

        defense_loader, pdefense_loader = load_cifar10(args, mode='defense')
        testset, ptestset = load_cifar10(args, mode='test')
        train1(model, args, defense_loader, testset, ptestset, unlearn=True, mask=False)
        train1(model, args, defense_loader, testset, ptestset, unlearn=False, mask=True, mask_penalty=False)
        from time import sleep
        sleep(1)
        save_mask_scores(model, "mask_scores.pth")
        # 假设mask_scores.pth文件在args.log_dir目录下
        mask_scores_path = args.log_dir / "mask_scores.pth"

        pruned_model_path = prune_with_scores(model, "mask_scores.pth", testset, ptestset, args.device)
        mask_values_path = os.path.join(app.config['SAVE_FOLDER'], "mask_values2.txt")

        # 返回处理结果
        # return jsonify({
        #     'progress': progress,
        #     'pruned_model_path': pruned_model_path
        # })
        return jsonify({
            'message': '存在后门风险的神经元已保存到"save"文件夹',
            'mask_values_url': url_for('download_file', filename='mask_values2.txt'),
            'pruned_model_path': mask_values_path
        })

        # return jsonify({
        #     'message': '存在后门风险的神经元已保存到"save"文件夹',
        #     'save_folder': app.config['SAVE_FOLDER'],
        #     'pruned_model_path': pruned_model_path
        # })

    return jsonify({'error': 'Invalid model file format'}), 400


@app.route('/save/<filename>')
def download_file(filename):
    return send_from_directory(app.config['SAVE_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER_UPLOAD'], exist_ok=True)
    os.makedirs(app.config['UPLOAD_FOLDER_RNP'], exist_ok=True)
    app.run(debug=True)

