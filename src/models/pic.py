
import torch.nn as nn
import torch 
import math
from compress.layers import ChannelMask
from .utils import conv, ste_round
from .base import CompressionModel
from .builder import define_encoder, define_hyperprior, define_decoder
from compress.entropy_models import GaussianConditional, EntropyBottleneck
# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))






class VarianceMaskingPIC(CompressionModel):

    def __init__(self, 
                N=192, 
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
                mask_policy = "point-based-std",
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
        self.mask_policy = mask_policy

        self.quality_list = [0,10]
        

        self.entropy_bottleneck = EntropyBottleneck(self.N)
        self.gaussian_conditional = GaussianConditional(None) #dddd

        self.masking = ChannelMask(self.mask_policy)
        self.num_slice_cumulative_list = [p//self.dim_chunk for p in self.dimensions_M]
        self.ns0 = self.num_slice_cumulative_list[0] 
        self.ns1 = self.num_slice_cumulative_list[1] 


        estremo_indice = self.support_progressive_slices + 1
        delta_dim = self.dimensions_M[1] - self.dimensions_M[0]


        self.g_a = define_encoder(self.multiple_encoder,N,M,self.dimensions_M)
        self.g_s = define_decoder(self.multiple_decoder,N,M,self.dimensions_M)

        self.h_a, self.h_mean_s, self.h_scale_s = define_hyperprior(self.multiple_hyperprior,
                                                                    self.M,
                                                                    self.N,
                                                                    self.dimensions_M)

        self.cc_mean_transforms = nn.ModuleList(
                nn.Sequential(
                    conv(self.division_dimension[0] + 32*min(i, 5), 224, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(224, 176, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(176, 128, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(128, 64, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(64, 32, stride=1, kernel_size=3),
                ) for i in range(self.num_slice_cumulative_list[0])
            )
        self.cc_scale_transforms = nn.ModuleList(
                nn.Sequential(
                    conv(self.division_dimension[0] + 32 * min(i, 5), 224, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(224, 176, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(176, 128, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(128, 64, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(64, 32, stride=1, kernel_size=3),
                ) for i in range(self.num_slice_cumulative_list[0])
                )
        self.lrp_transforms = nn.ModuleList(
                nn.Sequential(
                    conv(self.division_dimension[0] + 32 * min(i+1, 6), 224, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(224, 176, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(176, 128, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(128, 64, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(64, 32, stride=1, kernel_size=3),
                ) for i in range(self.num_slice_cumulative_list[0])
            )  


        pars_dimension = 32 
        self.cc_mean_transforms_prog = nn.ModuleList(
                    nn.Sequential(
                        conv(delta_dim + pars_dimension*min(i + 1, estremo_indice), 224, stride=1, kernel_size=3),
                        nn.GELU(),
                        conv(224, 176, stride=1, kernel_size=3),
                        nn.GELU(),
                        conv(176, 128, stride=1, kernel_size=3),
                        nn.GELU(),
                        conv(128, 64, stride=1, kernel_size=3),
                        nn.GELU(),
                        conv(64, 32, stride=1, kernel_size=3),
                    ) for i in range(self.num_slice_cumulative_list[1] - self.num_slice_cumulative_list[0] )
                )
        self.cc_scale_transforms_prog = nn.ModuleList(
                    nn.Sequential(
                        conv(delta_dim + pars_dimension*min(i + 1, estremo_indice), 224, stride=1, kernel_size=3),
                        nn.GELU(),
                        conv(224, 176, stride=1, kernel_size=3),
                        nn.GELU(),
                        conv(176, 128, stride=1, kernel_size=3),
                        nn.GELU(),
                        conv(128, 64, stride=1, kernel_size=3),
                        nn.GELU(),
                        conv(64, 32, stride=1, kernel_size=3),
                    ) for i in range(self.num_slice_cumulative_list[1]- self.num_slice_cumulative_list[0])
                    )

        self.lrp_transforms_prog = nn.ModuleList(
                nn.Sequential(
                    conv(delta_dim + 32 * min(i+2, estremo_indice + 1), 224, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(224, 176, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(176, 128, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(128, 64, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(64, 32, stride=1, kernel_size=3),
                ) for i in range(self.num_slice_cumulative_list[0])
            ) 


    def unfreeze_decoder(self):
        for n,p in self.named_parameters():
            p.requires_grad = False
        
        if self.multiple_decoder is False:
            for n,p in self.g_s.named_parameters():
                p.requires_grad = True
        else:
            for i in range(self.g_s):
                for n,p in self.g_s[i].named_parameters():
                    p.requires_grad = True


    def define_quality(self,quality):
        if quality is None:
            list_quality = self.quality_list
        elif isinstance(quality,list):
            if quality[0] == 0:
                list_quality = quality 
            else:
                list_quality = [0] + quality
        else:
            list_quality = [quality] 
        return list_quality


    def determine_support(self,y_hat_base,current_index,y_hat_quality):
        bi = y_hat_base[current_index]
        if current_index == 0 or self.support_progressive_slices == 0:
            return [bi]
        sup_ind = min(self.support_progressive_slices,current_index)
        psi_cum = y_hat_quality[current_index - sup_ind:current_index]
        return [bi] + psi_cum
    

    def merge(self,y_base,y_enhanced):
        return y_base + y_enhanced 
    


    def compute_hyperprior(self,y, quality = 10):

        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset
        if self.multiple_hyperprior is False or quality == 0:
            latent_scales = self.h_scale_s(z_hat) if self.multiple_hyperprior is False else self.h_scale_s[0](z_hat) 
            latent_means = self.h_mean_s(z_hat) if self.multiple_hyperprior is False else self.h_mean_s[0](z_hat)
            return latent_means, latent_scales, z_likelihoods
        else:
            latent_scales_base = self.h_scale_s[0](z_hat)
            latent_means_base = self.h_mean_s[0](z_hat)

            latent_scales_enh = self.h_scale_s[1](z_hat)
            latent_means_enh = self.h_mean_s[1](z_hat)

            latent_means = torch.cat([latent_means_base,latent_means_enh],dim = 1)
            latent_scales = torch.cat([latent_scales_base,latent_scales_enh],dim = 1)
            return latent_means, latent_scales, z_likelihoods


    def forward(self,x, quality = None, mask_pol = None, training = True):
    
        if mask_pol is None:
            mask_pol = self.mask_policy
        list_quality = self.define_quality(quality)  
        if self.multiple_encoder is False:
            y = self.g_a(x)
        else:
            y_base = self.g_a[0](x)
            y_enh = self.g_a[1](x)
            y = torch.cat([y_base,y_enh],dim = 1).to(x.device)
        y_shape = y.shape[2:]


        latent_means, latent_scales, z_likelihoods = self.compute_hyperprior(y, quality)
 
        y_slices = y.chunk(self.num_slices, 1) # total amount of slices

        y_hat_slices_base = []
        y_likelihood_base= []
        y_likelihood_enhanced = []
        x_hat_progressive = []

        scales_baseline = []

        mu_base, mu_prog = [],[]
        std_base, std_prog = [],[]


        for slice_index in range(self.ns0):
            y_slice = y_slices[slice_index]
            idx = slice_index%self.ns0

            indice = min(self.max_support_slices,idx)
            support_slices = (y_hat_slices_base if self.max_support_slices < 0 \
                                                        else y_hat_slices_base[:indice])               
            
            mean_support = torch.cat([latent_means[:,:self.dimensions_M[0]]] + support_slices, dim=1)
            scale_support = torch.cat([latent_scales[:,:self.dimensions_M[0]]] + support_slices, dim=1) 

            
            mu = self.cc_mean_transforms[idx](mean_support)  #self.extract_mu(idx,slice_index,mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]  
            mu_base.append(mu)

            scale = self.cc_scale_transforms[idx](scale_support)#self.extract_scale(idx,slice_index,scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]
            std_base.append(scale)

            scales_baseline.append(scale)
            _, y_slice_likelihood = self.gaussian_conditional(y_slice,
                                                            scale, 
                                                            mu, 
                                                            training = training)
            y_hat_slice = ste_round(y_slice - mu) + mu


            lrp_support = torch.cat([mean_support,y_hat_slice], dim=1)  ##ddd
            lrp = self.lrp_transforms[idx](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp               

            
            y_likelihood_enhanced.append(y_slice_likelihood)

            y_hat_slices_base.append(y_hat_slice)
            y_likelihood_base.append(y_slice_likelihood) 


        y_hat_b = torch.cat(y_hat_slices_base,dim = 1)

        x_hat_base = self.g_s[0](y_hat_b) if self.multiple_decoder else self.g_s(y_hat_b)



        x_hat_progressive.append(x_hat_base.unsqueeze(0))
        y_likelihoods_b = torch.cat(y_likelihood_base, dim=1)

        y_likelihood_total = []

        y_hat_total = []
        y_hat_total.append(y_hat_b)
        mu_total = []
        std_total = []
        


        for _,q in enumerate(list_quality[1:]):
            y_likelihood_quality = []
            y_likelihood_quality = y_likelihood_quality +  y_likelihood_base
            y_hat_slices_quality = [] 

            
            

            for slice_index in range(self.ns0,self.ns1):
                y_slice = y_slices[slice_index]

                current_index = slice_index%self.ns0
                
                if self.delta_encode:
                    y_slice = y_slice - y_slices[current_index] 

                support_vector_mu = mu_total if self.all_scalable else y_hat_slices_quality
                support_vector_std = std_total if self.all_scalable else y_hat_slices_quality

                support_slices_mean = self.determine_support(y_hat_slices_base, current_index,support_vector_mu)
                support_slices_std = self.determine_support(y_hat_slices_base,current_index,support_vector_std)
                              
                
                mean_support = torch.cat([latent_means[:,self.dimensions_M[0]:]] + support_slices_mean, dim=1)
                scale_support = torch.cat([latent_scales[:,self.dimensions_M[0]:]] + support_slices_std, dim=1) 

            
                mu = self.cc_mean_transforms_prog[current_index](mean_support)  #self.extract_mu(idx,slice_index,mean_support)
                mut = mu + y_hat_slices_base[current_index] if self.total_mu_rep else mu
                mu = mu[:, :, :y_shape[0], :y_shape[1]]  
                


                scale = self.cc_scale_transforms_prog[current_index](scale_support)#self.extract_scale(idx,slice_index,scale_support)

                std_total.append(scale)

                mu_total.append(mut)
                
                scale = scale[:, :, :y_shape[0], :y_shape[1]] #fff
                std_prog.append(scale)
                
                block_mask = self.masking(scale,pr = q,mask_pol = mask_pol) 

                
                block_mask = self.masking.apply_noise(block_mask, 
                                                      training if "learnable" in mask_pol else False)


                y_slice_m = y_slice  - mu
                y_slice_m = y_slice_m*block_mask

                _, y_slice_likelihood = self.gaussian_conditional(y_slice_m, 
                                                                  scale*block_mask, 
                                                                  training = training)
                y_hat_slice = ste_round(y_slice - mu)*block_mask + mu


                lrp_support = torch.cat([mean_support,y_hat_slice], dim=1)
                lrp = self.lrp_transforms_prog[current_index](lrp_support)
                lrp = 0.5 * torch.tanh(lrp)
                y_hat_slice += lrp   

                y_hat_slice = self.merge(y_hat_slice,y_hat_slices_base[current_index],current_index)
 
                y_hat_slices_quality.append(y_hat_slice)
                #y_hat_slices_only_quality.append(y_hat_slice)

                
                y_likelihood_quality.append(y_slice_likelihood)

            
            y_hat_enhanced = torch.cat(y_hat_slices_quality,dim = 1) 
            if self.multiple_decoder:
                x_hat_current = self.g_s[1](y_hat_enhanced)
            else: 
                #y_hat_enhanced = self.g_s_prog(y_hat_enhanced)
                x_hat_current = self.g_s(y_hat_enhanced)



            if self.u_net_post == 1:
                x_hat_current = self.refine(x_hat_current)
            elif self.u_net_post == 2:
                x_hat_current = self.refine[1](x_hat_current)


            y_likelihood_single_quality = torch.cat(y_likelihood_quality,dim = 1)
            y_likelihood_total.append(y_likelihood_single_quality.unsqueeze(0))
            x_hat_progressive.append(x_hat_current.unsqueeze(0)) #1,2,256,256

            y_hat_total.append(y_hat_enhanced)
        

        if len(y_likelihood_total)==0:
            y_likelihood_total = torch.ones_like(y_likelihoods_b).to(y_likelihoods_b.device)
        else:
            y_likelihood_total = torch.cat(y_likelihood_total,dim = 0)  #sliirrr
        x_hats = torch.cat(x_hat_progressive,dim = 0)

        return {
            "x_hat": x_hats,
            "likelihoods": {"y": y_likelihoods_b,"y_prog":y_likelihood_total,"z": z_likelihoods},
            "y_hat":y_hat_total,"y_base":y_hat_b,"y_prog":y_hat_enhanced,
            "mu_base":mu_base,"mu_prog":mu_prog,"std_base":std_base,"std_prog":std_prog
        }
    