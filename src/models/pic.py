
import torch.nn as nn
import torch 
import math
from compress.layers import ChannelMask

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))



class VarianceMaskingPIC(nn.Module):

    def __init__(self, N=192, 
                M=640, 
                division_dimension = [320,416],
                dim_chunk = 32,
                multiple_decoder = True,
                multiple_encoder = False,
                multiple_hyperprior = False,
                support_progressive_slices = 0,
                delta_encode = False,
                support_std = False,
                total_mu_rep = False,
                all_scalable = False,
                **kwargs):
        super().__init__(**kwargs)

        self.N = N
        self.M = M
        self.dim_chunk = dim_chunk 
        self.num_slices = int(M//self.dim_chunk) 
        self.multiple_encoder = multiple_encoder 
        self.multiple_decoder = multiple_decoder
        self.multiple_hyperprior = multiple_hyperprior
        self.division_channel = division_dimension[0]
        self.dimensions_M = division_dimension
        self.support_progressive_slices = support_progressive_slices
        self.delta_encode = delta_encode
        self.support_std = support_std
        self.total_mu_rep = total_mu_rep 
        self.all_scalable = all_scalable
        
