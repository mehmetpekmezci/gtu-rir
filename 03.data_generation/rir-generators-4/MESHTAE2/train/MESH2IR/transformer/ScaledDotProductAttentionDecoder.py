"""
    Scaled Dot-Product Attention implementation for the Decoder (from The Illustrated Transformer)
    From what I understood, this attention layer takes in Keys and Values matrices from the first_sublock_output
    of the encoder stack, so I have to modify the Scaled Dot-Product Attention implementation for the decoder.

    d_model ... embedding dimension
    d_k ... dimension of queries and keys
    d_v ... dimension of values
    masking ... a boolean indicating if look-ahead masking is to be applied (implemented based on https://medium.com/mlearning-ai/how-do-self-attention-masks-work-72ed9382510f)

    Q ... queries matrix (of dimensions number_of_samples x d_k)
    K ... keys matrix (of dimensions number_of_samples x d_k)
    V ... values matrix (of dimensions number_of_samples x d_v)
"""

import torch
import numpy as np
import math

class ScaledDotProductAttentionDecoder(torch.nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, masking=False):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.masking = masking

        self.embeddings_to_queries_layer = torch.nn.Linear(in_features=self.d_model, out_features=self.d_k)

        self.softmax_fn = torch.nn.Softmax(dim=-1)

    # TODO: maybe this function should have a better name since it returns True for all 2D shapes and beyond
    def is_matrix(self, arg):
        shape_of_arg = arg.shape
        shape_of_arg = list(shape_of_arg)
        if (len(shape_of_arg) > 1):
            return True
        else:
            return False

    def forward(self, embeddings, K, V):
        # TODO: an assert here that embeddings.shape[1] == self.d_model
        # TODO: an assert here that V.shape[1] == self.d_v

        Q = self.embeddings_to_queries_layer(embeddings)

        if (self.masking == False):
            if (self.is_matrix(K)):
                return torch.matmul(self.softmax_fn(torch.div(torch.matmul(Q, torch.transpose(K, -2, -1)), math.sqrt(self.d_k))), V)
            else:
                return torch.mul(self.softmax_fn(torch.div(torch.matmul(Q, K), math.sqrt(self.d_k))), V)
        else:
            # TODO: this could probably be written more efficiently
            first_dimension_of_M = Q.size(dim=0)
            second_dimension_of_M = K.size(dim=0) # first dimension of K because K gets transposed, so first dimension becomes the second dimension
            M = torch.zeros(first_dimension_of_M, second_dimension_of_M).cuda() # the look-ahead mask matrix
            for row_index in range(0, first_dimension_of_M):
                for column_index in range(row_index + 1, second_dimension_of_M):
                    M[row_index][column_index] = -float('inf')

            if (self.is_matrix(K)):
                return torch.matmul(self.softmax_fn(torch.div(torch.add(torch.matmul(Q, torch.transpose(K, -2, -1)), M), math.sqrt(self.d_k))), V)
            else:
                return self.softmax_fn(torch.div(torch.matmul(Q, K), math.sqrt(self.d_k))) * V
