from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

# Dataset name: flowers, birds
__C.DATASET_NAME = 'birds'
__C.EMBEDDING_TYPE = 'cnn-rnn'
__C.CONFIG_NAME = ''
__C.GPU_ID = '0'
__C.CUDA = True
__C.WORKERS = 6
__C.MAX_DIM= 30
__C.NET_G = ''
__C.NET_D = ''
__C.MESH_NET = ''
__C.STAGE1_G = ''
__C.DATA_DIR = ''
#__C.SYNTHETIC_APARTMENT_DATA_DIR = ''
__C.SYNTHETIC_GEOMETRIC_DATA_DIR = ''
__C.EVAL_DIR = ''
__C.VIS_COUNT = 32
__C.NUMBER_OF_TRANSFORMER_HEADS= 21
__C.NUMBER_OF_RIR_TRANSFORMER_HEADS=4
__C.MAX_FACE_COUNT= 2000
__C.Z_DIM = 100
__C.RIRSIZE = 4096
__C.MAX_CONNECTION_PER_NODE = 5
__C.STAGE = 1
__C.TRANSFORMER_VOCAB_SIZE=1000
__C.MESH_FACE_DATA_SIZE=16
__C.LATENT_VECTOR_SIZE=256
__C.FORCE_DECIMATION=True
__C.GWA_MESH_EMBEDDINGS_FILE='mesh_embeddings.pickle'
# Training options
__C.TRAIN = edict()
__C.PRE_TRAINED_MODELS_DIR=''
__C.MESH_NET_GAE_FILE=''
__C.TRAIN.FLAG = True
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.GAE_BATCH_SIZE = 64
__C.TRAIN.MAX_EPOCH = 600
__C.TRAIN.MAX_MESHNET_GAE_EPOCH = 600
__C.TRAIN.SNAPSHOT_INTERVAL = 50
__C.TRAIN.PRETRAINED_MODEL = ''
__C.TRAIN.PRETRAINED_EPOCH = 600
__C.TRAIN.LR_DECAY_EPOCH = 600
__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.MESH_LR = 2e-4

__C.TRAIN.COEFF = edict()
__C.TRAIN.COEFF.KL = 2.0

# Modal options
__C.GAN = edict()
__C.GAN.CONDITION_DIM = 128
__C.GAN.DF_DIM = 64
__C.GAN.GF_DIM = 128
__C.GAN.R_NUM = 4

__C.TEXT = edict()
__C.TEXT.DIMENSION = 1024


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f,Loader=yaml.Loader))

    _merge_a_into_b(yaml_cfg, __C)
