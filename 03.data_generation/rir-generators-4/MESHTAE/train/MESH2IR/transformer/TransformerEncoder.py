"""
    Transformer Encoder implementation (from Attention Is All You Need, section 3.1 and The Illustrated Transformer)

    d_model ... dimension of the embedding
    h ... number of parallel attention layers
    d_k ... dimension of queries and keys
    d_v ... dimension of values
    d_ff ... dimension of inner feedforward network layer
    number_of_encoder_blocks ... number of encoder blocks
"""

# not sure if this is the cleanest way to enable a module to be imported, but I found it on https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import sys
sys.path.insert(1, ".")

import torch
import numpy as np
from miscc.config import cfg

from transformer.EncoderBlock import EncoderBlock

class TransformerEncoder(torch.nn.Module):
    def __init__(self, d_model=512, h=8, d_k=64, d_v=64, d_ff=2048, number_of_encoder_blocks=6,d_latent_vector=cfg.LATENT_VECTOR_SIZE):
        super().__init__()

        self.d_model = d_model
        self.d_latent_vector=d_latent_vector
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff

        self.number_of_encoder_blocks = number_of_encoder_blocks

        self.encoder_blocks = torch.nn.ModuleList()
        for i in range(0, self.number_of_encoder_blocks):
            self.encoder_blocks.append(EncoderBlock(d_model=self.d_model, h=self.h, d_k=self.d_k, d_v=self.d_v, d_ff=self.d_ff))

        self.inner_autoencoder_encoder = torch.nn.Linear(in_features=cfg.MAX_FACE_COUNT*self.d_model, out_features=self.d_latent_vector)
        self.inner_autoencoder_decoder = torch.nn.Linear(in_features=self.d_latent_vector, out_features=cfg.MAX_FACE_COUNT*self.d_model)
        self.encoder_output_to_keys_layer = torch.nn.Linear(in_features=self.d_model, out_features=self.d_k)
        self.encoder_output_to_values_layer = torch.nn.Linear(in_features=self.d_model, out_features=self.d_v)

    def forward(self, embeddings):
        currentEncoderResult = embeddings

#        print(f"TransformerEncoder.embeddings.shape={embeddings.shape}")
        for encoderBlock in self.encoder_blocks:
            currentEncoderResult = encoderBlock(embeddings=currentEncoderResult)
        
        
        ### AUTOENCODING START
        org_shape=currentEncoderResult.shape
        currentEncoderResult=currentEncoderResult.reshape(-1,org_shape[1]*org_shape[2])
        Z=self.inner_autoencoder_encoder(currentEncoderResult)
        currentEncoderResult=self.inner_autoencoder_decoder(Z)
        currentEncoderResult=currentEncoderResult.reshape(-1,org_shape[1],org_shape[2])
        ### AUTOENCODING FINISH
#        Z=currentEncoderResult        
        
        K = self.encoder_output_to_keys_layer(currentEncoderResult)
        V = self.encoder_output_to_values_layer(currentEncoderResult)

        encoderOutput = Z

        return encoderOutput, K, V
