


import torch 
import time 
from .utils import extract_retrieve_entropy_parameters, decode_hyperprior


def decode_base(model, bits, latent_means, latent_scales, z_hat):
    y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
    y_string = bits
    y_hat_slices = []

    scales_base = []
    mu_base = []


    for slice_index in range(model.ns0): #ddd
        pr_strings = y_string[slice_index]
        idx = slice_index%model.ns0 #.num_slice_cumulative_list[0]
        indice = min(model.max_support_slices,idx)
        support_slices = (y_hat_slices if model.max_support_slices < 0 else y_hat_slices[:indice]) 
            
        mean_support = torch.cat([latent_means[:,:model.division_dimension[0]]] + support_slices, dim=1)
        scale_support = torch.cat([latent_scales[:,:model.division_dimension[0]]] + support_slices, dim=1) 

            
        mu = model.cc_mean_transforms[idx](mean_support)  #self.extract_mu(idx,slice_index,mean_support)
        mu = mu[:, :, :y_shape[0], :y_shape[1]]  
        scale = model.cc_scale_transforms[idx](scale_support)#self.extract_scale(idx,slice_index,scale_support)
        scale = scale[:, :, :y_shape[0], :y_shape[1]]

        scales_base.append(scale)
        mu_base.append(mu)

        index = model.gaussian_conditional.build_indexes(scale)


        rv = model.gaussian_conditional.decompress(pr_strings, index )
        rv = rv.reshape(mu.shape)
        y_hat_slice = model.gaussian_conditional.dequantize(rv, mu)

        lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
        lrp = model.lrp_transforms[idx](lrp_support)
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_slice += lrp

        #if quality == 0 or slice_index <self.num_slice_cumulative_list[0]:
        y_hat_slices.append(y_hat_slice)

    y_hat_b = torch.cat(y_hat_slices, dim=1)
    #x_hat = model.g_s[0](y_hat_b).clamp_(0, 1) if model.multiple_decoder else model.g_s(y_hat_b).clamp_(0, 1)
    scales = torch.cat(scales_base,dim = 1)
    mus = torch.cat(mu_base,dim = 1)
    #print("lo shape issss: ",y_hat_b.shape)
    return {"y_hat": y_hat_b, "scale":scales,"mu":mus}

def decode(model, 
           bitstreams,  
           q_ind = 0, 
           res_base = None,
           index_hat_slice = None,
           mean = None,
           z_data = None,
           entropy_data = None,
           y_checkpoints = None,
           rems = False
           ):

    q_list = bitstreams["q_list"]
    shape = bitstreams["shape"]
    z_string = bitstreams["z"] 
    base_string = bitstreams["base"]
    progressive_bits = bitstreams["progressive"]





    assert q_ind < len(q_list)
    if z_data is None:
        z_hat, latent_means, latent_scales, y_shape = decode_hyperprior(model, z_string,shape, q_ind)
        z_data = [z_hat,latent_means,latent_scales,y_shape]
    else:
        z_hat = z_data[0]
        latent_means = z_data[1]
        latent_scales = z_data[2] 
        y_shape = z_data[3]

    mu_total = []
    std_total = []

    #print("in input che succede: ",type(latent_scales))

    if res_base is None:
        res_base = decode_base(model,base_string, latent_means, latent_scales, z_hat)

    y_hat_base = res_base["y_hat"]
    mu_base = res_base["mu"]
    std_base = res_base["scale"]
    

        

    if q_ind == 0:
        #print("y hat base shape: ",y_hat_base.shape)
        x_hat = model.g_s[0](y_hat_base).clamp_(0, 1) if model.multiple_decoder else model.g_s(y_hat_base).clamp_(0, 1)
        return {"x_hat":x_hat,"y_hat":y_hat_base,"mu":mu_base,"scale":std_base}
    
    if y_checkpoints is not None:
        y_checkpoint_hat = [y.chunk(10,1) for y in y_checkpoints] #y_hat_base_slices  ###fff



    y_hat_slices_base = y_hat_base.chunk(10,1)
    mu_base = mu_base.chunk(10,1)
    std_base = std_base.chunk(10,1)

    if entropy_data is None:
        indexes_slices = []
        scale_slices = []
        mean = []
        mean_support_slices = []
        for slice_index in range(model.ns0,model.ns1):
            current_index = slice_index%model.ns0 #ddddddd
            mean_support, mu,mu_t, scale = extract_retrieve_entropy_parameters(current_index,
                                                            model,
                                                            mu_total,
                                                            std_total, 
                                                            y_hat_slices_base,
                                                            latent_means, 
                                                            latent_scales,
                                                            y_shape)

            if rems:
                if y_checkpoints is not None: 
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
                    

            indexes_sl = model.gaussian_conditional.build_indexes(scale).int() #[1,32,h,w]

            indexes_slices.append(indexes_sl)
            mu_total.append(mu_t)
            std_total.append(scale)
            mean.append(mu)
            scale_slices.append(scale)
            mean_support_slices.append(mean_support)

        mean = torch.cat(mean,dim = 1).squeeze(0)#.ravel() #[10,32,h,w]
        mean_total = torch.cat(mu_total,dim = 1).squeeze(0)
        index_hat_slice = torch.stack(indexes_slices).squeeze(1)#.ravel() #[10,32,h,w]
            
        entropy_data = [mean, mean_total, mean_support_slices, scale_slices,index_hat_slice]
    else: 
        mean = entropy_data[0]
        mean_total = entropy_data[1] 
        mean_support_slices = entropy_data[2] 
        scale_slices = entropy_data[3]
        index_hat_slice = entropy_data[4]
            

    
    M = model.division_channel
    

    h = y_shape[0]
    w = y_shape[1]
    means_elements = torch.zeros(M,h,w).float().to(mean.device)
    


    for j, qs in enumerate(q_list[:q_ind]):
        q_init = 0 if j == 0 else q_list[j-1]
        q_end=qs
        symbols = progressive_bits[j]


        delta_mask_i = model.masking.ProgMask(scale_slices,q_init)
        delta_mask_e = model.masking.ProgMask(scale_slices,q_end)

        delta_mask = delta_mask_e - delta_mask_i
        indexes_l = index_hat_slice*delta_mask
        
        symbols_q = model.gaussian_conditional.decompress(symbols,indexes_l) #[1,number of chosen elements]


        symbols_q  = symbols_q.reshape(M ,h,w)
        delta_mask = delta_mask.reshape(M ,h,w)
        means_elements += symbols_q*delta_mask #[1,10*32*h*w]

    means_elements += mean #*delta_mask
    output_dec = means_elements.float().reshape(1,M,h,w)
    r_output_slices =  output_dec.chunk(10,1)
    y_prog = []
    for slice_index in range(model.ns0,model.ns1):
        
        current_index = slice_index%model.ns0
        r = r_output_slices[current_index]
        lrp_support = torch.cat([mean_support_slices[current_index],r], dim=1)
        lrp = model.lrp_transforms_prog[current_index](lrp_support)
        lrp = 0.5 * torch.tanh(lrp)
        #print("lrp shape is ",lrp.shape)
        r +=    lrp
        
        r  = model.merge(r,y_hat_slices_base[current_index])
        y_prog.append(r) 

    y_prog = torch.cat(y_prog,1).reshape(1,M,h,w)

    x_hat = model.g_s[1](y_prog ) if model.multiple_decoder else model.g_s(y_prog)
    return {"x_hat": x_hat,
            "z_data":z_data,
            "entropy_data":entropy_data,
            "y_hat_base":y_hat_base,
            "y_prog":y_prog}   





