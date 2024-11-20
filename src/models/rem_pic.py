from .pic import VarianceMaskingPIC
import torch.nn as nn
from layers import LatentRateReduction
import torch
from .utils import ste_round
import time

class VarianceMaskingPICREM(VarianceMaskingPIC):

    def __init__(self, 
                N=192, 
                M=640, 
                division_dimension = [320,416],
                dim_chunk = 32,
                multiple_decoder = True,
                multiple_encoder = True,
                multiple_hyperprior = True,
                support_progressive_slices = 5,
                delta_encode = True,
                total_mu_rep = True,
                all_scalable = True,
                mask_policy = "point-based-std",
                check_levels = [0.01,0.25,1.75],
                mu_std = False,
                dimension = "big",
                **kwargs):
        super().__init__(N = N, 
                         M = M , 
                        division_dimension=division_dimension, 
                        dim_chunk=dim_chunk,
                        multiple_decoder=multiple_decoder,
                        multiple_encoder=multiple_encoder, 
                        multiple_hyperprior=multiple_hyperprior,
                        support_progressive_slices=support_progressive_slices,
                        delta_encode=delta_encode,
                        total_mu_rep=total_mu_rep,
                        all_scalable=all_scalable,
                        mask_policy=mask_policy,
                        **kwargs)
        
        self.dimension = dimension
        self.check_levels = check_levels 

        self.enable_rem = True # we start with enabling rems


        self.check_multiple = len(self.check_levels)
        self.mu_std = mu_std


        self.post_latent = nn.ModuleList(
                                nn.ModuleList( LatentRateReduction(dim_chunk = self.dim_chunk,
                                            mu_std = self.mu_std, dimension=dimension) 
                                            for _ in range(10))
                                for _ in range(self.check_multiple)
                                )


    def unfreeze_rems(self):
        for n,p in self.named_parameters():
            p.requires_grad = False

        for i in range(len(self.post_latent)):
            for n,p in self.post_latent[i].named_parameters():
                p.requires_grad = True      

    def load_state_dict(self, state_dict):
        # Carica i parametri del modello padre
        parent_state_dict = {k: v for k, v in state_dict.items() if k in self.state_dict() and 'post_latent' not in k}
        super().load_state_dict(parent_state_dict, strict=False)
        
        # Carica i parametri di post_latent se presenti
        post_latent_state_dict = {k: v for k, v in state_dict.items() if 'post_latent' in k}
        if post_latent_state_dict:
            print("I am downloading a model with REMS")
            self.post_latent.load_state_dict(post_latent_state_dict, strict=True)
        else:
            print("This model does not have trained REMs.  self.enable_rem will be set to False")
            self.enable_rem = False


    def print_information(self):

        if self.multiple_encoder is False:
            print(" g_a: ",sum(p.numel() for p in self.g_a.parameters()))
        else:
            print(" g_a: ",sum(p.numel() for p in self.g_a.parameters()))
           
        print(" h_a: ",sum(p.numel() for p in self.h_a.parameters()))
        print(" h_means_a: ",sum(p.numel() for p in self.h_mean_s.parameters()))
        print(" h_scale_a: ",sum(p.numel() for p in self.h_scale_s.parameters()))
        print("cc_mean_transforms",sum(p.numel() for p in self.cc_mean_transforms.parameters()))
        print("cc_scale_transforms",sum(p.numel() for p in self.cc_scale_transforms.parameters()))



        print("cc_mean_transforms_prog",sum(p.numel() for p in self.cc_mean_transforms_prog.parameters()))
        print("cc_scale_transforms_prog",sum(p.numel() for p in self.cc_scale_transforms_prog.parameters()))  

        print("lrp_transform",sum(p.numel() for p in self.lrp_transforms.parameters()))
        if self.multiple_decoder:
            for i in range(2):
                print("g_s_" + str(i) + ": ",sum(p.numel() for p in self.g_s[i].parameters()))
        else: 
            print("g_s",sum(p.numel() for p in self.g_s.parameters()))
        print("post net",sum(p.numel() for p in self.post_latent.parameters())) 
        print("post trainable net",sum(p.numel() for p in self.post_latent.parameters() if p.requires_grad is True)) 
        print("**************************************************************************")
        model_tr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_fr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad== False)
        print(" trainable parameters: ",model_tr_parameters)
        print(" freeze parameterss: ", model_fr_parameters)

        return sum(p.numel() for p in self.parameters() if p.requires_grad)






    def extract_chekpoint_representation_from_images(self,x, quality,  rc = True): #fff


        out_latent = self.compress( x, 
                                   quality =quality,
                                    mask_pol ="point-based-std",
                                    real_compress=rc) #["y_hat"] #ddd
            
        #if quality == self.check_levels[0]:
        return out_latent["y_hat"]
            
        out_latent_1 = self.compress( x, 
                                quality =self.check_levels[1],
                                mask_pol ="point-based-std",
                                checkpoint_rep= out_latent["y_hat"],
                                real_compress=rc)
            
        if quality == self.check_levels[1]:
            return out_latent_1["y_hat"]
            


        out_latent_2 = self.compress( x, 
                                quality =self.check_levels[2],
                                mask_pol ="point-based-std",
                                checkpoint_rep= out_latent_1["y_hat"],
                                real_compress=rc)
            
        return out_latent_2["y_hat"]




    def find_check_quality(self,quality):
        if quality <= self.check_levels[0]:
            quality_ref = 0 
            quality_post = 0

        elif (len(self.check_levels) == 2 or len(self.check_levels) == 3)  and self.check_levels[0] < quality <= self.check_levels[1]:
                quality_ref = self.check_levels[0]
                quality_post = self.check_levels[1]
        elif len(self.check_levels) == 2 and quality > self.check_levels[1]:
            quality_ref = self.check_levels[1]
            quality_post = 10
        
        elif len(self.check_levels)==3 and  self.check_levels[1] < quality <= self.check_levels[2]:
            quality_ref = self.check_levels[1] 
            quality_post = self.check_levels[-1]
        else:
            quality_ref = self.check_levels[-1]
            quality_post  = 10
        return quality_ref, quality_post



    def apply_latent_enhancement(self,
                                current_index,
                                block_mask,
                                bar_mask,
                                quality,
                                y_b_hat, 
                                mu_scale_base, 
                                mu_scale_enh,
                                mu, 
                                scale,
                                ):



        #bar_mask =   self.masking(scale,pr = quality_bar,mask_pol = mask_pol) 
        #star_mask = self.masking(scale,pr = quality,mask_pol = mask_pol)  

        attention_mask = block_mask - bar_mask 
        attention_mask = self.masking.apply_noise(attention_mask,  training = False)   

        if self.mu_std:
            attention_mask = torch.cat([attention_mask,attention_mask],dim = 1)  
        # in any case I do not improve anithing here!
        if quality <= self.check_levels[0]: #  in case nothing has to be done
            return mu, scale         

        if self.check_multiple == 1:
            enhanced_params =  self.post_latent[0][current_index](y_b_hat, mu_scale_base, mu_scale_enh, attention_mask)
        elif self.check_multiple == 2:
            index = 0 if self.check_levels[0] < quality <= self.check_levels[1] else 1 
            enhanced_params =  self.post_latent[index][current_index](y_b_hat, mu_scale_base, mu_scale_enh, attention_mask)
        else: 
            index = -1 
            if self.check_levels[0] < quality <= self.check_levels[1]: #ffff
                index = 0 
            elif  self.check_levels[1] < quality <= self.check_levels[2]:
                index = 1
            else:
                index = 2 
            enhanced_params =  self.post_latent[index][current_index](y_b_hat, mu_scale_base, mu_scale_enh, attention_mask)   
        if self.mu_std:
                mu,scale = enhanced_params.chunk(2,1)
                return mu, scale
        else:
            scale = enhanced_params
            return mu, scale



    def forward_single_quality(self, x, quality, mask_pol="point-based-std", training=False, checkpoint_ref = None):
        return self.forward(x = x, quality = quality, mask_pol = mask_pol, training = training, checkpoint_ref=checkpoint_ref)

    def forward(self, x, mask_pol = "point-based-std", quality = 0, training  = True, checkpoint_ref = None ):


        mask_pol = self.mask_policy if mask_pol is None else mask_pol

        if self.multiple_encoder is False:
            y = self.g_a(x)
            y_base = y 
            y_enh = y
        else:
            y_base = self.g_a[0](x)
            y_enh = self.g_a[1](x)
            y = torch.cat([y_base,y_enh],dim = 1).to(x.device) #dddd

        y_shape = y.shape[2:]
        latent_means, latent_scales, z_likelihoods = self.compute_hyperprior(y, quality)

        y_slices = y.chunk(self.num_slices, 1) # total amount of slicesy,

        y_hat_slices = []
        y_likelihood = []

        mu_base, mu_prog = [],[]
        std_base, std_prog = [],[]

        for slice_index in range(self.num_slice_cumulative_list[0]):
            y_slice = y_slices[slice_index]
            idx = slice_index%self.num_slice_cumulative_list[0]
            indice = min(self.max_support_slices,idx)
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:indice]) 
            
            mean_support = torch.cat([latent_means[:,:self.division_dimension[0]]] + support_slices, dim=1)
            scale_support = torch.cat([latent_scales[:,:self.division_dimension[0]]] + support_slices, dim=1) 

            
            mu = self.cc_mean_transforms[idx](mean_support)  #self.extract_mu(idx,slice_index,mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]  
            scale = self.cc_scale_transforms[idx](scale_support)#self.extract_scale(idx,slice_index,scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            mu_base.append(mu)
            std_base.append(scale) 

            mu_prog.append(mu) #le sommo
            std_prog.append(scale) #le sommo 

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu, training = training)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support,y_hat_slice], dim=1)
            lrp = self.lrp_transforms[idx](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp               

            y_hat_slices.append(y_hat_slice)
            y_likelihood.append(y_slice_likelihood)

        if quality == 0: #and  slice_index == self.num_slice_cumulative_list[0] - 1:
            y_hat = torch.cat(y_hat_slices,dim = 1)
            #x_hat = self.g_s[0](y_hat).clamp_(0, 1) if self.multiple_decoder else self.g_s(y_hat).clamp_(0, 1)
            y_likelihoods = torch.cat(y_likelihood, dim=1)
            return {
                "likelihoods": {"y": y_likelihoods,"z": z_likelihoods},
            "y_hat":y_hat,"y_base":y_hat,"y_complete":y_hat,
            "mu_base":mu_base,"mu_prog":mu_prog,"std_base":std_base,"std_prog":std_prog

            }         

        y_hat_b = torch.cat(y_hat_slices,dim = 1)
        y_hat_slices_quality = []
        

        y_likelihood_quality = []
        y_likelihood_quality = y_likelihood + []#ffff

        y_checkpoint_hat = checkpoint_ref.chunk(10,1) if checkpoint_ref is not None else y_hat_slices


        mu_total = []
        std_total = []

        for slice_index in range(self.ns0,self.ns1):

            y_slice = y_slices[slice_index]
            current_index = slice_index%self.ns0


            if self.delta_encode:
                y_slice = y_slice - y_slices[current_index] 


            support_vector = mu_total if self.all_scalable else y_hat_slices_quality
            support_vector_std = std_total if self.all_scalable else y_hat_slices_quality
            support_slices_mean = self.determine_support(y_hat_slices,
                                                         current_index,
                                                        support_vector                                                      
                                                         )
            support_slices_std = self.determine_support(y_hat_slices,
                                                         current_index,
                                                        support_vector_std                                                      
                                                         )

            mean_support = torch.cat([latent_means[:,self.division_dimension[0]:]] + support_slices_mean, dim=1)
            scale_support = torch.cat([latent_scales[:,self.division_dimension[0]:]] + support_slices_std, dim=1) 

            mu = self.cc_mean_transforms_prog[current_index](mean_support)  #self.extract_mu(idx,slice_index,mean_support)
            mut = mu + y_hat_slices[current_index] if self.total_mu_rep else mu
            mu = mu[:, :, :y_shape[0], :y_shape[1]]  

            scale = self.cc_scale_transforms_prog[current_index](scale_support)#self.extract_scale(idx,slice_index,scale_support)
            

            mu_prog[current_index] = mu_prog[current_index] + mu
            std_prog[current_index] = std_prog[current_index] +  scale 

            std_total.append(scale)
            mu_total.append(mut)

            scale = scale[:, :, :y_shape[0], :y_shape[1]] #fff


            # qua avviene la magia! 
            ms_base = torch.cat([mu_base[current_index],std_base[current_index]],dim = 1) 
            ms_progressive =  torch.cat([mu,scale],dim = 1) if self.mu_std else scale

            y_b_hat = y_checkpoint_hat[current_index]
            y_b_hat.requires_grad = True

            quality_bar, _  = self.find_check_quality(quality)

            block_mask =  self.masking(scale,pr = quality,mask_pol = mask_pol) # this is the q* in the original paper 
            block_mask = self.masking.apply_noise(block_mask, training)

            bar_mask =   self.masking(scale,pr = quality_bar,mask_pol = mask_pol)
            bar_mask = self.masking.apply_noise(bar_mask, training)

            
            if self.enable_rem:
                mu, scale = self.apply_latent_enhancement(current_index,
                                                        block_mask,
                                                        bar_mask,
                                                        quality,
                                                        y_b_hat, 
                                                        ms_base, 
                                                        ms_progressive,
                                                        mu, 
                                                        scale,
                                                        )
            

            y_slice_m = (y_slice  - mu)*block_mask
            _, y_slice_likelihood = self.gaussian_conditional(y_slice_m, scale*block_mask, training = training)
            y_hat_slice = ste_round(y_slice - mu)*block_mask + mu


            y_likelihood_quality.append(y_slice_likelihood)


            lrp_support = torch.cat([mean_support,y_hat_slice], dim=1)
            lrp = self.lrp_transforms_prog[current_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp   

            y_hat_slice = self.merge(y_hat_slice,y_hat_slices[current_index])   #ddd
            y_hat_slices_quality.append(y_hat_slice) 

        y_likelihoods = torch.cat(y_likelihood_quality,dim = 1) #ddddd
        y_hat_p = torch.cat(y_hat_slices_quality,dim = 1) 

        mu_base = torch.cat(mu_base,dim = 1)
        mu_prog = torch.cat(mu_prog,dim = 1)
        std_base = torch.cat(std_base,dim = 1)   
        std_prog = torch.cat(std_prog, dim = 1)#kkkk

        index = 0 if quality == 0 else 1
        x_hat = self.g_s[index](y_hat).clamp_(0, 1) if self.multiple_decoder  \
                else self.g_s(y_hat).clamp_(0, 1)

        return {
            "x_hat":x_hat,
            "likelihoods": {"y": y_likelihoods,"z": z_likelihoods},
            "y_hat":y_hat_p,"y_base":y_hat_b,
            "mu_base":mu_base,"mu_prog":mu_prog,"std_base":std_base,"std_prog":std_prog
        }     


    def compress(self, x, 
                quality = 0.0, 
                mask_pol = "point-based-std", 
                checkpoint_rep = None, 
                real_compress = True 
                ):

        #used_qual = self.check_levels if used_qual is None else used_qual


        if self.multiple_encoder is False:
            y = self.g_a(x)
        else:
            y_base = self.g_a[0](x)
            y_enh = self.g_a[1](x)
            y = torch.cat([y_base,y_enh],dim = 1).to(x.device)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings =  self.entropy_bottleneck.compress(z)
        
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        latent_scales = self.h_scale_s(z_hat) if self.multiple_hyperprior is False \
                        else self.h_scale_s[0](z_hat)
        latent_means = self.h_mean_s(z_hat) if self.multiple_hyperprior is False \
                        else self.h_mean_s[0](z_hat)


        if self.multiple_hyperprior and quality > 0:
            latent_scales_enh = self.h_scale_s[1](z_hat) 
            latent_means_enh = self.h_mean_s[1](z_hat)
            latent_means = torch.cat([latent_means,latent_means_enh],dim = 1)
            latent_scales = torch.cat([latent_scales,latent_scales_enh],dim = 1) 

        y_hat_slices = []

        y_slices = y.chunk(self.num_slices, 1) # total amount of slices
        y_strings = []
        masks = []
        mu_base = []
        std_base = [] 


        for slice_index in range(self.ns0):
            y_slice = y_slices[slice_index]
            indice = min(self.max_support_slices,slice_index%self.ns0)


            support_slices = (y_hat_slices if self.max_support_slices < 0 \
                                                        else y_hat_slices[:indice])               
            
            idx = slice_index%self.ns0


            mean_support = torch.cat([latent_means[:,:self.division_dimension[0]]] + support_slices, dim=1)
            scale_support = torch.cat([latent_scales[:,:self.division_dimension[0]]] + support_slices, dim=1) 

            
            mu = self.cc_mean_transforms[idx](mean_support)  #self.extract_mu(idx,slice_index,mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]  
            scale = self.cc_scale_transforms[idx](scale_support)#self.extract_scale(idx,slice_index,scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]


            mu_base.append(mu) 
            std_base.append(scale)

            index = self.gaussian_conditional.build_indexes(scale)
            if real_compress:
                y_q_string  = self.gaussian_conditional.compress(y_slice, index,mu)
                y_hat_slice = self.gaussian_conditional.decompress(y_q_string, index).to(mu.device)
                #print(y_hat_slice.device,"  ",mu.device)
                y_hat_slice = y_hat_slice + mu
            else:
                y_q_string  = self.gaussian_conditional.quantize(y_slice, "symbols", mu)#ddd
                y_hat_slice = y_q_string + mu

            y_strings.append(y_q_string)

            lrp_support = torch.cat([mean_support,y_hat_slice], dim=1)
            lrp = self.lrp_transforms[idx](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        if quality <= 0:
            return {"strings": [y_strings, z_strings],
                    "shape":z.size()[-2:], 
                    "masks":masks,
                    "y":y,
                    "y_hat":torch.cat(y_hat_slices,dim = 1),
                    "latent_means":latent_means,
                    "latent_scales":latent_scales,
                    "mean_base":torch.cat(mu_base,dim = 1),
                    "std_base":torch.cat(std_base,dim = 1)}
        

        y_hat_slices_quality = []

        y_b_hats = checkpoint_rep.chunk(10,1) if checkpoint_rep is not None else y_hat_slices 

        mu_total, std_total = [],[]
        for slice_index in range(self.ns0,self.ns1): #ffff

            y_slice = y_slices[slice_index]
            current_index = slice_index%self.ns0

            if self.delta_encode:
                y_slice = y_slice - y_slices[current_index] 


            support_vector = mu_total if self.all_scalable else y_hat_slices_quality
            support_vector_std = std_total if self.all_scalable else y_hat_slices_quality
            support_slices_mean = self.determine_support(y_hat_slices,
                                                         current_index,
                                                        support_vector                                                      
                                                         )
            support_slices_std = self.determine_support(y_hat_slices,
                                                         current_index,
                                                        support_vector_std                                                      
                                                         )


            
            mean_support = torch.cat([latent_means[:,self.division_dimension[0]:]] + support_slices_mean, dim=1)
            scale_support = torch.cat([latent_scales[:,self.division_dimension[0]:]] + support_slices_std, dim=1) 

            mu = self.cc_mean_transforms_prog[current_index](mean_support)  #self.extract_mu(idx,slice_index,mean_support)
            mut = mu + y_hat_slices[current_index] if self.total_mu_rep else mu
            mu = mu[:, :, :y_shape[0], :y_shape[1]]  

            scale = self.cc_scale_transforms_prog[current_index](scale_support)#self.extract_scale(idx,slice_index,scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]] #fff

            std_total.append(scale)
            mu_total.append(mut)

            y_b_hat = y_b_hats[current_index]

            ms_base = torch.cat([mu_base[current_index],std_base[current_index]],dim = 1) 
            ms_progressive =  torch.cat([mu,scale],dim = 1) if self.mu_std else scale
            
            
            
            quality_bar,quality_post = self.find_check_quality(quality)


            block_mask = self.masking(scale,pr = quality,mask_pol = mask_pol)
            block_mask = self.masking.apply_noise(block_mask, False)
            masks.append(block_mask)

            bar_mask = self.masking(scale,pr = quality_bar,mask_pol = mask_pol)
            bar_mask = self.masking.apply_noise(bar_mask, False)


            if self.enable_rem:
                mu, scale = self.apply_latent_enhancement(current_index,
                                                        block_mask,
                                                        bar_mask,
                                                        quality,
                                                        y_b_hat, 
                                                        ms_base, 
                                                        ms_progressive,
                                                        mu, 
                                                        scale,
                                                        )
                
            index = self.gaussian_conditional.build_indexes(scale*block_mask).int()
            if real_compress:

                y_q_string  = self.gaussian_conditional.compress((y_slice - mu)*block_mask, index)
                y_strings.append(y_q_string)
                y_hat_slice_nomu = self.gaussian_conditional.quantize((y_slice - mu)*block_mask, "dequantize") 
                y_hat_slice = y_hat_slice_nomu + mu
            else:
                y_q_string  = self.gaussian_conditional.quantize((y_slice - mu)*block_mask, "dequantize") 
                y_strings.append(y_q_string)
                y_hat_slice = y_q_string + mu



            lrp_support = torch.cat([mean_support,y_hat_slice], dim=1)
            lrp = self.lrp_transforms_prog[current_index](lrp_support) #ddd
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slice = self.merge(y_hat_slice,y_hat_slices[current_index])
            y_hat_slices_quality.append(y_hat_slice)
        
        return {"strings": [y_strings, z_strings],"shape":z.size()[-2:],"masks":masks,"y_hat":torch.cat(y_hat_slices_quality,dim = 1)}
    

    def decompress(self, 
                   strings,
                    shape, 
                    quality,
                    mask_pol = None,
                    checkpoint_rep = None, 
                    ):
        


        mask_pol = self.mask_policy if mask_pol is None else mask_pol


        start_t = time.time()


        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat) if self.multiple_hyperprior is False    \
                        else self.h_scale_s[0](z_hat)
        latent_means = self.h_mean_s(z_hat) if self.multiple_hyperprior is False \
                        else self.h_mean_s[0](z_hat)

    
        if self.multiple_hyperprior and quality > 0:
            latent_scales_enh = self.h_scale_s[1](z_hat) 
            latent_means_enh = self.h_mean_s[1](z_hat)
            latent_means = torch.cat([latent_means,latent_means_enh],dim = 1)
            latent_scales = torch.cat([latent_scales,latent_scales_enh],dim = 1) 


        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        y_string = strings[0]
        y_hat_slices = []


        mu_base = []
        std_base = []



        for slice_index in range(self.num_slice_cumulative_list[0]): #ddd
            pr_strings = y_string[slice_index]
            idx = slice_index%self.num_slice_cumulative_list[0]
            indice = min(self.max_support_slices,idx)
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:indice]) 
            
            mean_support = torch.cat([latent_means[:,:self.division_dimension[0]]] + support_slices, dim=1)
            scale_support = torch.cat([latent_scales[:,:self.division_dimension[0]]] + support_slices, dim=1) 
      
            mu = self.cc_mean_transforms[idx](mean_support)  #self.extract_mu(idx,slice_index,mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]  
            scale = self.cc_scale_transforms[idx](scale_support)#self.extract_scale(idx,slice_index,scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            mu_base.append(mu)
            std_base.append(scale)

            index = self.gaussian_conditional.build_indexes(scale)


            rv = self.gaussian_conditional.decompress(pr_strings, index )
            rv = rv.reshape(mu.shape)
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[idx](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        if quality == 0:
            y_hat_b = torch.cat(y_hat_slices, dim=1)

            

            end_t = time.time()
            time_ = end_t - start_t
 
            x_hat = self.g_s[0](y_hat_b).clamp_(0, 1) if self.multiple_decoder else \
                    self.g_s(y_hat_b).clamp_(0, 1)
            return {"x_hat": x_hat, "y_hat": y_hat_slices,"time":time_}


        start_t = time.time()

        y_hat_slices_quality = []


        y_b_hats = checkpoint_rep.chunk(10,1) if checkpoint_rep is not None else y_hat_slices  
        mu_total,std_total = [],[]

        
        for slice_index in range(self.ns0,self.ns1):
            pr_strings = y_string[slice_index]
            current_index = slice_index%self.ns0

            support_slices_mean = self.determine_support(y_hat_slices,
                                                         current_index,
                                                        mu_total if self.all_scalable else y_hat_slices_quality                                                      
                                                         )
            support_slices_std = self.determine_support(y_hat_slices,
                                                         current_index,
                                                        std_total if self.all_scalable else y_hat_slices_quality                                                     
                                                         )


            
            mean_support = torch.cat([latent_means[:,self.division_dimension[0]:]] + support_slices_mean, dim=1)
            scale_support = torch.cat([latent_scales[:,self.division_dimension[0]:]] + support_slices_std, dim=1) 

            mu = self.cc_mean_transforms_prog[current_index](mean_support)  #self.extract_mu(idx,slice_index,mean_support)
            mut = mu + y_hat_slices[current_index] if self.total_mu_rep else mu
            mu = mu[:, :, :y_shape[0], :y_shape[1]]  

            scale = self.cc_scale_transforms_prog[current_index](scale_support)#self.extract_scale(idx,slice_index,scale_support)
            
            std_total.append(scale)
            mu_total.append(mut)

            scale = scale[:, :, :y_shape[0], :y_shape[1]] #fff

            y_b_hat =  y_b_hats[current_index]

            #y_b_hat = y_b_hats[current_index]
            ms_base = torch.cat([mu_base[current_index],std_base[current_index]],dim = 1) 
            ms_progressive =  torch.cat([mu,scale],dim = 1) if self.mu_std else scale


            
                   
            quality_bar,quality_post = self.find_check_quality(quality)

            block_mask = self.masking(scale,pr = quality,mask_pol = mask_pol) 
            bar_mask = self.masking(scale,pr = quality_bar,mask_pol = mask_pol)
            
            post_mask = self.masking(scale,pr = quality_post,mask_pol = mask_pol) 



            if self.enable_rem:
                mu, scale = self.apply_latent_enhancement(current_index,
                                                        post_mask,
                                                        bar_mask,
                                                        quality,
                                                        y_b_hat, 
                                                        ms_base, 
                                                        ms_progressive,
                                                        mu, 
                                                        scale,
                                                        )


            
 


            index = self.gaussian_conditional.build_indexes(scale*block_mask)
            rv = self.gaussian_conditional.decompress(pr_strings, index).to(mu.device)
            rv = rv.reshape(mu.shape)

            y_hat_slice = rv + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms_prog[current_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slice = self.merge(y_hat_slice,y_hat_slices[current_index])

            y_hat_slices_quality.append(y_hat_slice)

        y_hat_en = torch.cat(y_hat_slices_quality,dim = 1)


        end_t = time.time()
        time_ = end_t - start_t

        if self.multiple_decoder:
            x_hat = self.g_s[1](y_hat_en).clamp_(0, 1)
        else:
            x_hat = self.g_s(y_hat_en).clamp_(0, 1) 
        return {"x_hat": x_hat,"y_hat":y_hat_en,"time":time_}          

