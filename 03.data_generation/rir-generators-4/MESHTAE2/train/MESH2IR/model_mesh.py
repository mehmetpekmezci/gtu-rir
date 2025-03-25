import torch
import torch.nn as nn
import torch.nn.parallel
from miscc.config import cfg
#from miscc.utils import mask_test_edges
from torch.autograd import Variable
import numpy as np
from torch_geometric.nn import GCNConv, GATConv,TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import to_dense_adj, to_dense_batch, dropout_adj,unbatch

import torch.nn.functional as F
import scipy.sparse as sp

import traceback
from torchinfo import summary
import math
#import subprocess

import torch.nn.functional as F
from einops import repeat
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
#### bu net degil : from chamfer_dist import ChamferDistanceL1
#from chamferdist import ChamferDistance
import copy
import numpy as np
import math
from functools import partial

import torch
import timm.models.vision_transformer

from timm.models.vision_transformer import PatchEmbed, Block

from transformer.TransformerEncoder import TransformerEncoder
#from transformer.TransformerDecoder import TransformerDecoder
from transformer.MultiHeadAttention import MultiHeadAttention
from transformer.EncoderDecoderAttention import EncoderDecoderAttention
from transformer.FeedForward import FeedForward

class MESH_TRANSFORMER_ENCODER(nn.Module):
        def __init__(self):
                super(MESH_TRANSFORMER_ENCODER,self).__init__()
                
                #self.LATENT_VECTOR_SIZE=128
                
                self.EMBEDDING_DIM=9 # triangle corners coordinates (9= v1.xyz, v2.xyz, v3.xyz) , 3+3+3=9
                self.CENTER_NORMAL_AREA_DIM=7 # center coords (3), Normal Vector(3),Area (1),  3+3+1 = 7
                self.positional_embedding_predictor_linear_layer_1=torch.nn.Linear(self.CENTER_NORMAL_AREA_DIM,cfg.LATENT_VECTOR_SIZE)
                self.positional_embedding_predictor_linear_layer_2=torch.nn.Linear(cfg.LATENT_VECTOR_SIZE,self.EMBEDDING_DIM)
                
#                self.gaussian_matrix=torch.rand((self.EMBEDDING_DIM, self.EMBEDDING_DIM_V), dtype=torch.float).cuda()
                
                
                self.loss_fn = nn.MSELoss()#torch.nn.CrossEntropyLoss()
                
                ## MP : h == 8 == number of parallel heads (multiheads)
                self.transformer_encoder = TransformerEncoder(
                                              d_model=self.EMBEDDING_DIM, h=cfg.NUMBER_OF_TRANSFORMER_HEADS, d_k=self.EMBEDDING_DIM, d_v=self.EMBEDDING_DIM, d_ff=512, number_of_encoder_blocks=1,
                                              d_latent_vector=cfg.LATENT_VECTOR_SIZE 
                                              )

        @torch.autocast(device_type="cuda",dtype=torch.float16)         
        def encode(self,embeddings):

#                 print(f"MESH_TRANSFORMER_ENCODER.embeddings.shape={embeddings.shape}")
                 
                 Z, K, V = self.transformer_encoder(embeddings)
 
                 return Z,K,V


        @torch.autocast(device_type="cuda",dtype=torch.float16)         
        def normal_plus_positional_embeddings(self,X):
            
               normal_embeddings=X[:,:,:self.EMBEDDING_DIM] ## first nine elements are v1.xyz, v2xyz, v3.xyz
               
               ## last 7 elements are center.xyz, normal.xyz, and area, we convert it to 9 elements using neural network.
               positional_embedding_0=torch.relu(self.positional_embedding_predictor_linear_layer_1(X[:,:,self.EMBEDDING_DIM:])) 
               positional_embeddings=torch.relu(self.positional_embedding_predictor_linear_layer_2(positional_embedding_0))            
            

               embeddings=normal_embeddings+positional_embeddings
               
               
               
               return embeddings
                 
        @torch.autocast(device_type="cuda",dtype=torch.float16)         
        def forward(self,X):
#                print(f"MESH_TRANSFORMER_ENCODER.forward.X.shape={X.shape}")
                embeddings=self.normal_plus_positional_embeddings(X)
                Z, K, V = self.encode(embeddings)
                return Z,K,V,embeddings
        

class MESH_TRANSFORMER_DECODER_MULTI_HEAD_ATTENTION(nn.Module):
        def __init__(self):
                super(MESH_TRANSFORMER_DECODER_MULTI_HEAD_ATTENTION,self).__init__()
                self.EMBEDDING_DIM=9 # triangle corners coordinates (9= v1.xyz, v2.xyz, v3.xyz) , 3+3+3=9
                self.add_and_norm_layer = torch.nn.LayerNorm(normalized_shape=self.EMBEDDING_DIM)
                self.multi_head_attention_layer = MultiHeadAttention(d_model=self.EMBEDDING_DIM, h=cfg.NUMBER_OF_TRANSFORMER_HEADS, d_k=self.EMBEDDING_DIM, d_v=self.EMBEDDING_DIM, masking=False)

        @torch.autocast(device_type="cuda",dtype=torch.float16)         
        def forward(self,tokenEmbedding,K,V):
            multi_head_output = self.add_and_norm_layer(tokenEmbedding + self.multi_head_attention_layer(tokenEmbedding))
            return multi_head_output              


class MESH_TRANSFORMER_DECODER_ENCODER_DECODER_ATTENTION(nn.Module):
        def __init__(self):
                super(MESH_TRANSFORMER_DECODER_ENCODER_DECODER_ATTENTION,self).__init__()
                self.EMBEDDING_DIM=9 # triangle corners coordinates (9= v1.xyz, v2.xyz, v3.xyz) , 3+3+3=9
                self.loss_fn = nn.MSELoss()#torch.nn.CrossEntropyLoss()
                self.encoder_decoder_attention_layer = EncoderDecoderAttention(d_model=self.EMBEDDING_DIM, h=cfg.NUMBER_OF_TRANSFORMER_HEADS, d_k=self.EMBEDDING_DIM, d_v=self.EMBEDDING_DIM, masking=False)
                self.add_and_norm_layer = torch.nn.LayerNorm(normalized_shape=self.EMBEDDING_DIM)
                self.feed_forward_layer = FeedForward(d_model=self.EMBEDDING_DIM, d_ff=512)
                self.decoder_output_to_logits_layer = torch.nn.Linear(in_features=self.EMBEDDING_DIM, out_features=cfg.TRANSFORMER_VOCAB_SIZE)

                 
        @torch.autocast(device_type="cuda",dtype=torch.float16)         
        def forward(self,tokenEmbedding,K,V,multi_head_output):
            encoder_decoder_attention_output = self.add_and_norm_layer(multi_head_output + self.encoder_decoder_attention_layer(tokenEmbedding, K, V))
            norm = self.add_and_norm_layer(encoder_decoder_attention_output + self.feed_forward_layer(encoder_decoder_attention_output))
            logits = self.decoder_output_to_logits_layer(norm)
            return logits              

        @torch.autocast(device_type="cuda",dtype=torch.float16)         
        def loss(self,Y_pred,Y) :
                return self.loss_fn(Y_pred,Y)





'''
class MESH_TRANSFORMER_AE(nn.Module):
        def __init__(self):
                super(MESH_TRANSFORMER_AE,self).__init__()
                
                #self.LATENT_VECTOR_SIZE=128
                
                self.EMBEDDING_DIM=9 # triangle corners coordinates (9= v1.xyz, v2.xyz, v3.xyz) , 3+3+3=9
                self.CENTER_NORMAL_AREA_DIM=7 # center coords (3), Normal Vector(3),Area (1),  3+3+1 = 7
                self.positional_embedding_predictor_linear_layer_1=torch.nn.Linear(self.CENTER_NORMAL_AREA_DIM,cfg.LATENT_VECTOR_SIZE)
                self.positional_embedding_predictor_linear_layer_2=torch.nn.Linear(cfg.LATENT_VECTOR_SIZE,self.EMBEDDING_DIM)
                
#                self.gaussian_matrix=torch.rand((self.EMBEDDING_DIM, self.EMBEDDING_DIM_V), dtype=torch.float).cuda()
                
                
                self.loss_fn = nn.MSELoss()#torch.nn.CrossEntropyLoss()
                
                ## MP : h == 8 == number of parallel heads (multiheads)
                self.transformer_encoder = TransformerEncoder(
                                              d_model=self.EMBEDDING_DIM, h=cfg.NUMBER_OF_TRANSFORMER_HEADS, d_k=self.EMBEDDING_DIM, d_v=self.EMBEDDING_DIM, d_ff=512, number_of_encoder_blocks=1,
                                              d_latent_vector=cfg.LATENT_VECTOR_SIZE 
                                              )
                self.transformer_decoder = TransformerDecoder(
                                               vocab_size=cfg.TRANSFORMER_VOCAB_SIZE, 
                                               d_model=self.EMBEDDING_DIM, h=cfg.NUMBER_OF_TRANSFORMER_HEADS, d_k=self.EMBEDDING_DIM, d_v=self.EMBEDDING_DIM, d_ff=512, number_of_decoder_blocks=1
                                            , masking=False)
                
        def encode(self,embeddings):

#                 print(f"MESH_TRANSFORMER_AE.encoder.embeddings.shape={embeddings.shape}")
                 
                 Z, K, V = self.transformer_encoder(embeddings)
 
                 return Z,K,V


        def normal_plus_positional_embeddings(self,X):
            
               normal_embeddings=X[:,:,:9] ## first nine elements are v1.xyz, v2xyz, v3.xyz
               
               ## last 7 elements are center.xyz, normal.xyz, and area, we convert it to 9 elements using neural network.
               positional_embedding_0=torch.relu(self.positional_embedding_predictor_linear_layer_1(X[:,:,9:])) 
               positional_embeddings=torch.relu(self.positional_embedding_predictor_linear_layer_2(positional_embedding_0))            
            

               embeddings=normal_embeddings+positional_embeddings
               
               
               
               return embeddings
                 
                 
        def forward(self,X):
#                print(f"MESH_TRANSFORMER_AE.forward.X.shape={X.shape}")
                
                embeddings=self.normal_plus_positional_embeddings(X)
               
                Z, K, V = self.encode(embeddings)
                
#                print(f"Z.shape={Z.shape} K.shape={K.shape}")

                Y=embeddings ## AUTOENCODER
                Y_PREDICTED  = self.decode(Y, K, V)
                return Y_PREDICTED,Z

        def decode(self,Y, K, V):
                return self.transformer_decoder(Y, K, V)
              
        def loss(self,Y_pred,Y) :
                return self.loss_fn(Y_pred,Y)

'''              

