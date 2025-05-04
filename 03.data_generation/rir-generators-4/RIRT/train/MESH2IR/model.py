import torch
import torch.nn as nn
import torch.nn.parallel
from miscc.config import cfg
#from miscc.utils import mask_test_edges
from torch.autograd import Variable
import numpy as np
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import to_dense_adj, to_dense_batch, dropout_adj
import torch.nn.functional as F
import scipy.sparse as sp

import traceback
from torchinfo import summary
#from torch_geometric.nn.models import  GAE,InnerProductDecoder
#from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
#from torch_scatter import scatter
from transformer.TransformerEncoder1 import TransformerEncoder1
from transformer.TransformerDecoder import TransformerDecoder



class RIR_TRANSFORMER(nn.Module):
        def __init__(self):
                super(RIR_TRANSFORMER,self).__init__()

                self.EMBEDDING_DIM=6 # spk-mic coordinates (6= spk.xyz, mic.xyz) , 3+3=6
                self.MESH_EMBED_DIM=cfg.LATENT_VECTOR_SIZE # 
                self.positional_embedding_predictor_linear_layer_1=torch.nn.Linear(self.MESH_EMBED_DIM,cfg.RIRSIZE)
                self.positional_embedding_predictor_linear_layer_2=torch.nn.Linear(self.EMBEDDING_DIM,cfg.RIRSIZE)
                
#                self.gaussian_matrix=torch.rand((self.EMBEDDING_DIM, self.D_K_V), dtype=torch.float).cuda()
                
                
                self.loss_fn = nn.MSELoss()#torch.nn.CrossEntropyLoss()
                
                ## MP : h == 8 == number of parallel heads (multiheads)
                self.transformer_encoder = TransformerEncoder1(
                                               #d_model=self.EMBEDDING_DIM,h=cfg.NUMBER_OF_TRANSFORMER_HEADS, d_k=self.EMBEDDING_DIM, d_v=self.EMBEDDING_DIM, d_ff=512, number_of_encoder_blocks=2)
                                               d_model=cfg.RIRSIZE,h=cfg.NUMBER_OF_RIR_TRANSFORMER_HEADS, d_k=cfg.RIRSIZE, d_v=cfg.RIRSIZE, d_ff=512, number_of_encoder_blocks=1)
                self.transformer_decoder = TransformerDecoder(
                                               #vocab_size=cfg.TRANSFORMER_VOCAB_SIZE, 
                                               vocab_size=cfg.RIRSIZE, 
                                               #d_model=self.EMBEDDING_DIM, 
                                               d_model=cfg.RIRSIZE,
                                               h=cfg.NUMBER_OF_RIR_TRANSFORMER_HEADS, d_k=cfg.RIRSIZE, d_v=cfg.RIRSIZE, d_ff=512, number_of_decoder_blocks=1
                                            , masking=False)
                
        def encode(self,embeddings):

#                 print(f"MESH_TRANSFORMER_AE.encoder.embeddings.shape={embeddings.shape}")
                 
                 K, V = self.transformer_encoder(embeddings)
 
                 return K,V


        def normal_plus_positional_embeddings(self,text_embedding,mesh_embed):
               positional_embeddings=torch.relu(self.positional_embedding_predictor_linear_layer_1(mesh_embed)) 
               text_embeddings=torch.relu(self.positional_embedding_predictor_linear_layer_2(text_embedding)) 
               embeddings=text_embeddings+positional_embeddings
               return embeddings
                 
                 
        def forward(self, text_embedding,mesh_embed):
                #full_embed= torch.cat((mesh_embed, text_embedding), 1)
                        
                embeddings=self.normal_plus_positional_embeddings(text_embedding,mesh_embed)
               
                K, V = self.encode(embeddings)

                Y_PREDICTED  = self.decode(embeddings, K, V)
                return Y_PREDICTED

        def decode(self,Y, K, V):
                return self.transformer_decoder(Y, K, V)
              
        def loss(self,Y_pred,Y) :
                Y=Y[:,:,:cfg.RIRSIZE]
                Y_pred=torch.unsqueeze(Y_pred, 0)
                Y_pred=Y_pred[:,:,:cfg.RIRSIZE]
                return self.loss_fn(Y_pred,Y)

              




