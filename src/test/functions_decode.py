


import torch 
import time 
from .utils import extract_retrieve_entropy_parameters, extract_latents_from_bits
q_list = [0.002,0.05,0.5,0.75,1,1.5,2,2.5,3,4,5,5.5,6,6.6,10] 


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

    return {"y_hat": y_hat_b, "scale":scales,"mu":mus}

def decode(model, bitstreams, shape, q_ind = 0, y_hat_base = None):

    q_list = bitstreams["q_list"]
    assert q_ind < len(q_list)

    latent_means, latent_scales, y_shape = extract_latents_from_bits(model, bitstreams, q_ind)


    mu_total = []
    std_total = []



    if y_hat_base is not None:
        res_base = decode_base(model, latent_means, latent_scales, bitstreams["base"])
        y_hat_base = res_base["y_hat"]

    if q_ind == 0:
        x_hat = model.g_s[0](y_hat_base).clamp_(0, 1) if model.multiple_decoder else model.g_s(y_hat_base).clamp_(0, 1)
        return {"x_hat":x_hat,"y_hat":y_hat_base}
    


    y_hat_slices_base = y_hat_base.chunk(10,1)


    indexes_slices = []
    scale_slices = []
    mean = []
    mean_support_slices = []
    for slice_index in range(model.ns0,model.ns1):

        current_index = slice_index%model.ns0
        mean_support, mu, mu_t, scale = extract_retrieve_entropy_parameters(current_index,
                                                            model,
                                                            mu_total,
                                                            std_total, 
                                                            y_hat_slices_base,
                                                            latent_means, 
                                                            latent_scales,
                                                            y_shape)
        mean_support_slices.append(mean_support)
        mean.append(mu.ravel().unsqueeze(0))
        mu_total.append(mu_t)
        std_total.append(scale)
        indexes_sl = model.gaussian_conditional.build_indexes(scale).int() #[1,ch,h,w] 
        indexes_sl = indexes_sl.ravel().unsqueeze(0) #[1,1*ch*h*w]

        scale_slice = scale.ravel().unsqueeze(0) #[1,1*ch*h*w]

        indexes_slices.append(indexes_sl)
        scale_slices.append(scale_slice)


    index_hat_slice = torch.cat(indexes_slices,dim = 0) #[10,ch*h*w]
    scale_hat_slice = torch.cat(scale_slices,dim = 0) # [10,ch*h*w]
    mean = torch.cat(mean,dim = 0)

    #ora inizia la parte di decoding 
    ordered_scale,std_ordering_index = torch.sort(scale_hat_slice, dim=1, descending=True) 
    ordered_index = torch.gather(index_hat_slice,dim = 1,index = std_ordering_index) #[10,ch*h*w]
    
    
    inverse_indices = torch.argsort(std_ordering_index, dim=1)
    
    ordered_mean = torch.gather(mean,dim =1, index = std_ordering_index )
    r_dec = ordered_mean
    r_decode = []


    for j in range(q_ind):
        qs =q_list[j]

        q_end = qs*10
        q_init = 0 if j == 0 else q_list[j-1]
        q_init = q_init*10
        init_length = int((q_init*shape)/100) + 1 #if j > 0 else 0 
        end_length =  int((q_end*shape)/100) 

        symbols = bitstreams[j]
        ordered_index_bc = ordered_index[:,init_length:end_length + 1] #ddd
        indexes_l = torch.flatten(ordered_index_bc).unsqueeze(0)


        symbols_q = model.gaussian_conditional.decompress(symbols,
                                                             indexes_l,
                                                             already_quantize = True) #[1,10*(end_length - init_length)]
        
        symbols_q = symbols_q.to("cuda")
        r_hat_tbc = symbols_q.reshape(1,10,-1)  #[1,10,(end_length - init_length)]

        r_decode.append(r_hat_tbc)
    
    r_decode = torch.cat(r_decode,dim = 1) # [10, Ndecoded]
    r_shape = r_decode.shape[1]

    r_dec[:, :r_shape] = r_decode #[10, ch*w*h]

    r_dec = torch.gather(r_dec, dim=1, index=inverse_indices) #[10,ch*w*h]

    r_dec = r_dec.reshape(10,y_shape[0],y_shape[1])#questo dovrebbe essere tutto!!!

    y_hat_slices = []
    for slice_index in range(model.ns0,model.ns1):
        current_index = slice_index%model.ns0
        y_hat_slice = r_dec[current_index]


        mean_support = mean_support_slices[current_index]        
        lrp_support = torch.cat([mean_support,y_hat_slice], dim=1)
        lrp = model.lrp_transforms_prog[current_index](lrp_support)
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_slice += lrp   

        y_hat_slice = model.merge(y_hat_slice,y_hat_slices_base[current_index])
 
        y_hat_slices.append(y_hat_slice)
    

    y_hat_en = torch.cat(y_hat_slices,dim = 1)
    if model.multiple_decoder:
        x_hat = model.g_s[1](y_hat_en)
    else:
        x_hat = model.g_s(y_hat_en)
    return {"x_hat": x_hat}   





