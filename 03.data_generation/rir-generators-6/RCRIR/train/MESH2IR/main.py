from __future__ import print_function
import torch.backends.cudnn as cudnn
import torch
import torchvision.transforms as transforms

import pickle
import random
import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil
import dateutil.tz

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

from miscc.datasets import RIRDataset
from miscc.config import cfg, cfg_from_file
from miscc.utils import mkdir_p
from trainer import GANTrainer
from  torch.utils.data import DataLoader
import glob


def load_embedding(data_dir,embedding_file_name):
        print("Loading embeddings ...")
        embedding_directory   = data_dir+'/'+embedding_file_name
        with open(embedding_directory, 'rb') as f:
            embeddings = pickle.load(f)
        print(embedding_file_name+" embeddings are loaded ...")
        return embeddings


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='birds_stage1.yml', type=str)
    parser.add_argument('--gpu',  dest='gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    num_gpu = len(cfg.GPU_ID.split(','))
    if cfg.TRAIN.FLAG:
        embeddings = load_embedding(cfg.DATA_DIR,'training.embeddings.pickle')
        print(f"len(embeddings)={len(embeddings)}")
        rir_dataset = RIRDataset(cfg.DATA_DIR, embeddings, rirsize=cfg.RIRSIZE)
        assert rir_dataset
        print(f"batch_size of rir dataloaader : {cfg.TRAIN.BATCH_SIZE * num_gpu}")
        data_loader=DataLoader(rir_dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu,num_workers=int(cfg.WORKERS),shuffle=True,drop_last=True)
        print(f"len(data_loader): {len(data_loader)}")
        algo = GANTrainer(output_dir)
        algo.train(data_loader,stage=cfg.STAGE)
