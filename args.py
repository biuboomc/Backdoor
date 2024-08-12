import argparse
import yaml

import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import logging
import os
BASE_DIR = Path(__file__).parent
print(BASE_DIR)
def init_and_load_args(path):
    args = init_args()
    add_yaml_to_args(args, path)
    preprocess_args(args)
    return args

def init_args():
    parser = argparse.ArgumentParser(description='vit')
    args = parser.parse_args()
    return args

def add_yaml_to_args(args, path):
    print('[add {} to args]'.format(path))
    with open(path, "r") as f:
        mix_defaults = yaml.safe_load(f.read())
    print(mix_defaults)
    mix_defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
    args.__dict__ = mix_defaults

def preprocess_args(args):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    set_seed(args.seed)
    get_log_folder_name(args)
    args.logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, 'output.log')),
            logging.StreamHandler()
        ])
    if args.device == 'auto':
        if torch.cuda.is_available():
            args.device = 'cuda'
        else:
            args.device = 'cpu'
    if 'cuda' in args.device:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if args.num_workers == 'auto':
        args.num_workers = int(os.cpu_count() * 0.8)

    args.num_classes = 10
    args.input_size = [224,224]


def set_seed(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_log_folder_name(args):
    now = datetime.now()
    folder_name = "-".join(
        [
            args.dataset,
            args.model,
            args.attack,
            now.strftime("%m%d%H%M")
        ]
    )
    args.log_dir = BASE_DIR / "logs" / folder_name
    if not Path.exists(args.log_dir):
        Path.mkdir(args.log_dir, parents=True)