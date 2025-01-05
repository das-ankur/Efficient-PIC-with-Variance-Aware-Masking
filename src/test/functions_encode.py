

from utility import  read_image, compute_padding
import torch.nn.functional as F 
import os
import pickle
import itertools
import time
import torch
from .utils import extract_retrieve_entropy_parameters 


q_list = [0.002,0.05,0.5,0.75,1,1.5,2,2.5,3,4,5,5.5,6,6.6] 

def encode(model, x_padded,
           save_path= None,
           rems = False,
           q_list = q_list,
           y_checkpoints = None):


    print("encode level base")

    out_base = model.compress(x_padded, quality = 0)  
    


    mu_base = out_base["mean_base"]
    std_base = out_base["scale_base"]
    y_hat_base = out_base["y_hat_base"]


    bitstreams = {}
    bitstreams["q_list"] = q_list
    bitstreams["shape"] = out_base["shape"]
    bitstreams["z"] = out_base["strings"][1]
    bitstreams["base"] = out_base["strings"][0]
    
    bits_z =  sum(len(s) for s in bitstreams["z"]) * 8.0 
    bits_base = sum(len(s[0]) for s in bitstreams["base"]) * 8.0


    start_t = time.time()
    bitstreams_list, bits_prog = extract_all_bitsreams(model,
                                        x_padded,
                                        y_hat_base,
                                        mu_base,
                                        std_base,
                                        q_list,
                                        rems = rems,
                                        y_checkpoints = y_checkpoints)

    end_t = time.time()
    bitstreams["progressive"] = bitstreams_list 
    #bitstreams["y_shape"] = c 
    #bitstreams["unpad"] = unpad
    print("Done encoding PP: ",end_t-start_t) 
    if save_path is not None:
        os.makedirs(save_path,exist_ok = True)
        name =  os.path.join(save_path,"bits.pkl")
        with open(name, 'wb') as file:
           pickle.dump(bitstreams, file)
    
    return bitstreams,[bits_z,bits_base,bits_prog]



def extract_all_bitsreams(model, 
                          x, 
                          y_hat_base, 
                          mu_base,
                          std_base, 
                          q_list,
                          y_checkpoints = None, 
                          rems = False):


    if model.multiple_encoder:
        y_base = model.g_a[0](x)
        y_enh = model.g_a[1](x)
        y = torch.cat([y_base,y_enh],dim = 1).to(x.device)
    else:
        y = model.g_a(x)
    y_shape = y.shape[2:]

    latent_means, latent_scales, _ = model.compute_hyperprior(y) #dddd

    y_hat_slices_base = y_hat_base.chunk(10,1)
    mu_total = [] #if mu_total is None else mu_total
    std_total = [] #if std_total is None else std_total


    y_slices = y.chunk(model.num_slices, 1)
    if y_checkpoints is not None:
        y_checkpoint_hat = [y.chunk(10,1) for y in y_checkpoints] #y_hat_base_slices  ###fff



    mu_base = mu_base.chunk(10,1)
    std_base = std_base.chunk(10,1)

 
    r_slices = []
    indexes_slices = []
    scale_slices = []

    mean = []
    mean_support_slices = []

    for slice_index in range(model.ns0,model.ns1):

        current_index = slice_index%model.ns0 
        y_slice = y_slices[slice_index]
        if model.delta_encode:
            r_slice = y_slice - y_slices[current_index]


        mean_support, mu,mu_t, scale = extract_retrieve_entropy_parameters(current_index,
                                                            model,
                                                            mu_total,
                                                            std_total, 
                                                            y_hat_slices_base,
                                                            latent_means, 
                                                            latent_scales,
                                                            y_shape)
        
        if rems:
            print("here I am checking if I need to apply rem anhancement")
            assert len(y_checkpoints)== model.num_rems 
            for j in range(len(y_checkpoints)):
                
                y_b_hat = y_checkpoint_hat[j][current_index]
                ms_base = torch.cat([mu_base[current_index],std_base[current_index]],dim = 1) 
                ms_progressive =  torch.cat([mu,scale],dim = 1) if model.mu_std else scale
                y_b_hat.requires_grad = True
                mu, scale = model.apply_latent_enhancement(current_index,
                                                            model.check_levels[j] ,
                                                            model.check_levels[j + 1] if j < model.num_rems - 1 else 10,
                                                            y_b_hat, 
                                                            ms_base, 
                                                            ms_progressive,
                                                            mu, 
                                                            scale
                                                            )
           
        mu_total.append(mu_t)
        std_total.append(scale)
        mean.append(mu)
        mean_support_slices.append(mean_support)



        r_slice_quantize = model.gaussian_conditional.quantize(r_slice - mu,"symbols") #[1,32,h,w]
        indexes_sl = model.gaussian_conditional.build_indexes(scale).int() #[1,32,h,w]
        
        
        r_slices.append(r_slice_quantize)
        indexes_slices.append(indexes_sl)
        scale_slices.append(scale)


    mean = torch.cat(mean,dim = 1).squeeze(0)#.ravel() #[10,32,h,w]
    #prova = torch.cat(prova,dim = 1).squeeze(0)
    #print("dimensione della prova",prova.shape)

    r_hat_slice = torch.stack(r_slices).squeeze(1)#.ravel() #[10,32,h,w]
    index_hat_slice = torch.stack(indexes_slices).squeeze(1)#.ravel() #[10,32,h,w]

    bitstream = []
    bits = []

    for j, qs in enumerate(q_list):



        q_init = 0 if j == 0 else q_list[j-1]
        q_end = qs


        delta_mask_i = model.masking.ProgMask(scale_slices,q_init)#.ravel() #[10,320,h,w] of ones and zeros
        delta_mask_e = model.masking.ProgMask(scale_slices,q_end)#.ravel() #[320,h,w] of ones and zeros


            
        
        
        delta_mask = delta_mask_e - delta_mask_i#.bool()

        indexes_l = index_hat_slice*delta_mask
        symbols_list =  r_hat_slice*delta_mask
        
        symbols_q = model.gaussian_conditional.compress(symbols_list,indexes_l,already_quantize = True)
        bpp_scale = sum(len(s) for s in symbols_q) * 8.0 
        
        bits.append(bpp_scale)
        bitstream.append(symbols_q)

    return bitstream,bits





