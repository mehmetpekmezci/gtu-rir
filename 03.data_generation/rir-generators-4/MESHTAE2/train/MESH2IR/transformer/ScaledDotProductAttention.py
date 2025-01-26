"""
    Scaled Dot-Product Attention implementation (from Attention Is All You Need, section 3.2.1)

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
#import subprocess

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, masking=False):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.masking = masking

        self.embeddings_to_queries_layer = torch.nn.Linear(in_features=self.d_model, out_features=self.d_k)
        self.embeddings_to_keys_layer = torch.nn.Linear(in_features=self.d_model, out_features=self.d_k)
        self.embeddings_to_values_layer = torch.nn.Linear(in_features=self.d_model, out_features=self.d_v)

        self.softmax_fn = torch.nn.Softmax(dim=-1)

    # TODO: maybe this function should have a better name since it returns True for all 2D shapes and beyond
    def is_matrix(self, arg):
        shape_of_arg = arg.shape
        shape_of_arg = list(shape_of_arg)
        if (len(shape_of_arg) > 1):
            return True
        else:
            return False

    def forward(self, embeddings):
        # TODO: an assert here that embeddings.shape[1] == self.d_model

#        print(f"ScaledDotProductAttention.embeddings.shape={embeddings.shape}")
#        print(f"self.embeddings_to_queries_layer={self.embeddings_to_queries_layer}")
        #subprocess.run(["nvidia-smi"])
        #print("ScaledDotProductAttention_1############")
        Q = self.embeddings_to_queries_layer(embeddings)
        #subprocess.run(["nvidia-smi"])
        #print("ScaledDotProductAttention_2############")
        K = self.embeddings_to_keys_layer(embeddings)
        #subprocess.run(["nvidia-smi"])
        #print("ScaledDotProductAttention_3############")
        V = self.embeddings_to_values_layer(embeddings)
        #subprocess.run(["nvidia-smi"])
        #print("ScaledDotProductAttention_4############")

        if (self.masking == False):
            if (self.is_matrix(K)):
                tK= torch.transpose(K, -2, -1)
                result0= torch.matmul(Q, tK)
                #subprocess.run(["nvidia-smi"])
                #print(f"ScaledDotProductAttention_5.1.1############ Q.shape={Q.shape} K.shape={K.shape} tK.shape={tK.shape}")
                result11= math.sqrt(self.d_k)
                #subprocess.run(["nvidia-smi"])
                #print(f"ScaledDotProductAttention_5.1.1.1############ result0.shape={result0.shape} result11={result11}")
                result12= result0
                #result12= torch.div(result0, result11)
                #subprocess.run(["nvidia-smi"])
                #print(f"ScaledDotProductAttention_5.1.1.2############")
                result1= result12
                #result1= self.softmax_fn(result12)
                #subprocess.run(["nvidia-smi"])
                #print(f"ScaledDotProductAttention_5.1.2############ result1.shape={result1.shape} V.shape={V.shape} ")
                result= torch.matmul(result1, V)
                #result= torch.matmul(self.softmax_fn(torch.div(torch.matmul(Q, torch.transpose(K, -2, -1)), math.sqrt(self.d_k))), V)
                #subprocess.run(["nvidia-smi"])
                #print(f"ScaledDotProductAttention_5.1.3############ result.shape={result.shape}")
                return result
            else:
                result= torch.mul(self.softmax_fn(torch.div(torch.matmul(Q, K), math.sqrt(self.d_k))), V)
                #subprocess.run(["nvidia-smi"])
                #print("ScaledDotProductAttention_5.2############")
                return result
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
