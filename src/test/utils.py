
from utility import  read_image, compute_padding
import torch 
import torch.nn.functional as F 


def read_and_pads_image(d,device):
    x = read_image(d).to(device)
    x = x.unsqueeze(0) 
    h, w = x.size(2), x.size(3)
    pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
    x_padded = F.pad(x, pad, mode="constant", value=0)
    return x_padded, unpad


def extract_latents_from_bits(model,bitstreams,q_ind):
    z_hat = model.entropy_bottleneck.decompress(bitstreams["z"], bitstreams["shape"])

    y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
    latent_scales_base = model.h_scale_s(z_hat) if model.multiple_hyperprior is False else model.h_scale_s[0](z_hat)
    latent_means_base = model.h_mean_s(z_hat) if model.multiple_hyperprior is False else model.h_mean_s[0](z_hat)

    if model.multiple_hyperprior is False or q_ind == 0:
        latent_means =  latent_means_base # torch.cat([latent_means_base,latent_means_enh],dim = 1)
        latent_scales = latent_scales_base #torch.cat([latent_scales_base,latent_scales_enh],dim = 1) 
    else:
        latent_scales_enh = model.h_scale_s[1](z_hat) 
        latent_means_enh = model.h_mean_s[1](z_hat)
        latent_means = torch.cat([latent_means_base,latent_means_enh],dim = 1)
        latent_scales = torch.cat([latent_scales_base,latent_scales_enh],dim = 1) 
    return latent_means, latent_scales, y_shape

def extract_retrieve_entropy_parameters(current_index,model,mu_total,std_total, y_hat_slices_base,latent_means, latent_scales, y_shape):
    


    support_slices_mean = model.determine_support(y_hat_slices_base, current_index, mu_total)
    support_slices_std = model.determine_support(y_hat_slices_base, current_index, std_total)


            
    mean_support = torch.cat([latent_means[:,model.division_dimension[0]:]] + support_slices_mean, dim=1)
    scale_support = torch.cat([latent_scales[:,model.division_dimension[0]:]] + support_slices_std, dim=1) 

    mu = model.cc_mean_transforms_prog[current_index](mean_support)  #self.extract_mu(idx,slice_index,mean_support)
    mut = mu + y_hat_slices_base[current_index] if model.total_mu_rep else mu
    mu = mu[:, :, :y_shape[0], :y_shape[1]]  

    scale = model.cc_scale_transforms_prog[current_index](scale_support)#self.extract_scale(idx,slice_index,scale_support)
    scale = scale[:, :, :y_shape[0], :y_shape[1]] #fff


    return mean_support, mu, mut, scale 