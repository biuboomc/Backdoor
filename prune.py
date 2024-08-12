from tqdm import tqdm
from configs import BASE_DIR
from configs import init_and_load_args
from dataset import load_cifar10, test
import torch
from vit import TiViT
from pathlib import Path


def clip_masks(model, threshold=1.0):
    masks = model.get_masks()
    with torch.no_grad():
        for mask in masks:
            mask.data.clamp_(0, threshold)


def l1_penalty(model):
    masks = model.get_masks()
    res = 0.
    for mask in masks:
        res += mask.abs().sum()
    return res


def train(model, args, trainloader, test_loader=None, p_test_loader=None, mask=False, unlearn=False,
          mask_penalty=False):
    if mask:
        optimizer = torch.optim.SGD(model.get_masks(), lr=args.lr_mask)
    else:
        optimizer = torch.optim.SGD(model.get_params(), lr=args.lr)
        mask_penalty = False
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    for i in range(1, args.epochs + 1):
        model.train()
        acc = 0.
        total_loss = 0.
        for data in tqdm(trainloader):
            x, y = data
            x, y = x.to(args.device), y.to(args.device)
            print(f"Shape of x: {x.shape}")  # 打印 x 的形状
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


def save_mask_scores(model, filename="mask_scores.pth"):
    mask_dict = dict()
    for key, value in model.state_dict().items():
        if 'mask' in key:
            mask_dict[key] = value
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 保存掩码字典到指定的文件路径
    torch.save(mask_dict, log_dir / filename)
    return mask_dict

def prune_with_scores(model, mask_scores, testloader, ptestloader, device, interval = 20):
    if isinstance(mask_scores, str):
        mask_scores = torch.load(args.log_dir/mask_scores)
    print(mask_scores)
    scores = []
    for key, value in mask_scores.items():
        value = value.squeeze()
        for i in range(len(value)):
            scores.append((key, i, value[i]))
    scores = sorted(scores, key=lambda x: x[2])
    # print(scores)
    # results = []
    for i, neuron in enumerate(scores):
        key, idx, score = neuron

        state_dict = model.state_dict()
        state_dict[key][0,0,0,0,idx] = 0.
        model.load_state_dict(state_dict, strict = False)
        if i %interval == 0:
            ca, loss = test(model, testloader, device)
            asr, ploss = test(model, ptestloader, device)
            print("{}, idx: {}, mask: {:.4f}, ca: {:.4f}, loss: {:.4f}, asr: {:.4f}, ploss: {:.4f}".format(key, idx, score,ca, loss, asr, ploss))
    with open('./save/mask_values2.txt', "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask Score \n')
        for i, neuron in enumerate(scores):
            key, idx, score = neuron

            f.write(f"{i} \t {key} \t {idx} \t {score:.4f} \n")
# def prune_with_scores(model, mask_scores, testloader, ptestloader, device, interval=20, threshold =0.4):
#     if isinstance(mask_scores, str):
#         mask_scores = torch.load(args.log_dir / mask_scores)
#     # print(mask_scores)
#     scores = []
#     for key, value in mask_scores.items():
#         value = value.squeeze()
#         for i in range(len(value)):
#             scores.append((key, i, value[i]))
#     scores = sorted(scores, key=lambda x: x[2])
#     print(scores)
#     prune_neurons = []
#     for i, neuron in enumerate(scores):
#         key, idx, score = neuron
#         # state_dict = model.state_dict()
#         if score <= threshold:
#             prune_neurons.append((key, idx))
#     for neuron in prune_neurons:
#         key, idx = neuron
#         state_dict = model.state_dict()
#         state_dict[key][0, 0, 0, 0, idx] = 0.  # 将对应的掩膜值设置为0
#         model.load_state_dict(state_dict, strict=False)  # 更新模型中的参数
#         # if i % interval == 0:  # 每隔一定间隔测试模型性能
#         ca, loss = test(model, testloader, device)  # 测试模型在正常数据上的性能
#         asr, ploss = test(model, ptestloader, device)  # 测试模型在包含后门攻击的数据上的性能
#         print("{}, idx: {}, mask: {:.4f}, ca: {:.4f}, loss: {:.4f}, asr: {:.4f}, ploss: {:.4f}".format(key, idx,
#                                                                                                        score, ca,
#                                                                                                        loss, asr,
#                                                                                                        ploss))

        #在这个修改后的版本中，我们首先创建了一个布尔列表prune_neurons，用于存储所有分数低于threshold的神
        # if i % interval == 0:
        #     ca, loss = test(model, testloader, device)
        #     asr, ploss = test(model, ptestloader, device)
        #     print("{}, idx: {}, mask: {:.4f}, ca: {:.4f}, loss: {:.4f}, asr: {:.4f}, ploss: {:.4f}".format(key, idx,
        #                                                                                                    score, ca,
        #                                                                                                    loss, asr,
        #                                                                                                    ploss))


if __name__ == '__main__':
    args = init_and_load_args(BASE_DIR / 'configs/prune.yaml')
    logger = args.logger
    logger.info('----------- Load Model {} to Device {} --------------'.format(args.model, args.device))
    logger.info('----------- Load Dataset {} with Attack {} --------------'.format(args.dataset, args.attack))

    # model = resnet18(num_classes = args.num_classes).to(args.device)
    model = TiViT(args.num_classes, masked=True).to(args.device)
    state_dict = torch.load('./anp/10_0.9660_1.00_badnet.pth')
    model.load_state_dict(state_dict, strict=False)

    defense_loader, pdefense_loader = load_cifar10(args, mode='defense')
    testset, ptestset = load_cifar10(args, mode='test')

    train(model, args, defense_loader, testset, ptestset, unlearn=True, mask=False)
    train(model, args, defense_loader, testset, ptestset, unlearn=False, mask=True, mask_penalty=True)
    save_mask_scores(model, "mask_scores_rnp.pth")
    model.reset_masks()
    prune_with_scores(model, "mask_scores_rnp.pth", testset, ptestset, args.device)