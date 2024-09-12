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

from miscc.datasets import TextDataset,MeshDataset
from miscc.config import cfg, cfg_from_file
from miscc.utils import mkdir_p
from trainer import GANTrainer
from gae_trainer import GAETrainer
#from torch_geometric.loader import DataLoader
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
    parser.add_argument('--synthetic_geometric_data_dir', dest='synthetic_geometric_data_dir', type=str, default='')
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
    if args.synthetic_geometric_data_dir != '':
        cfg.SYNTHETIC_GEOMETRIC_DATA_DIR = args.synthetic_geometric_data_dir
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
    output_dir = '../output/%s_%s_%s' % \
                 (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    num_gpu = len(cfg.GPU_ID.split(','))
    if cfg.TRAIN.FLAG:
        embeddings = load_embedding(cfg.DATA_DIR,'embeddings.pickle')
        synthetic_geometric_embeddings=load_embedding(cfg.SYNTHETIC_GEOMETRIC_DATA_DIR,'synthetic_geometric_embeddings.pickle')
        print(f"len(embeddings)={len(embeddings)}  len(synthetic_geometric_embeddings)={len(synthetic_geometric_embeddings)}")

        mesh_dataset = MeshDataset(cfg.SYNTHETIC_GEOMETRIC_DATA_DIR, synthetic_geometric_embeddings) #,augment=["scale","deformation"])      
        assert mesh_dataset
        mesh_only_train_data_loader=DataLoader(mesh_dataset, batch_size=cfg.TRAIN.GAE_BATCH_SIZE * num_gpu, num_workers=int(cfg.WORKERS),)
        gaeTrainer = GAETrainer(output_dir)

        if not os.path.exists(cfg.PRE_TRAINED_MODELS_DIR+"/"+cfg.MESH_NET_GAE_FILE):
           print("GAE MESH NET PRETARINED MODEL DOES NOT EXISTS SO STARTING TO TRAIN THE GAE_MESH_NET MODEL ......")
           gaeTrainer.train(mesh_only_train_data_loader, cfg.STAGE)
           print("GAE training is finished, now we are training GAN for IR ......")
        else :
           print("There exists a pre-trained GAE MESH NET model, GANTrainer will load it ......")

        rir_dataset = TextDataset(cfg.DATA_DIR, embeddings, rirsize=cfg.RIRSIZE)
        assert rir_dataset
        rir_and_mesh_train_data_loader=DataLoader(rir_dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu,num_workers=int(cfg.WORKERS),) #shuffle=True

        algo = GANTrainer(output_dir,gaeTrainer.mesh_net)
        algo.train(rir_and_mesh_train_data_loader,cfg.STAGE)
    # else:
    #     file_path = cfg.EVAL_DIR
    #     algo = GANTrainer(output_dir)
    #     algo.sample(file_path, cfg.STAGE)
