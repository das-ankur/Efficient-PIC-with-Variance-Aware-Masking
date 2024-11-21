

from utility import  read_image, compute_padding
import torch.nn.functional as F 
import os
import pickle
import time
import torch
from .utils import extract_retrieve_entropy_parameters 


q_list = [0.002,0.05,0.5,0.75,1,1.5,2,2.5,3,4,5,5.5,6,6.6,10] 

def encode(model,d,device,savepath,rems = False,q_list = q_list):

    nome_immagine = d.split("/")[-1].split(".")[0]
    x = read_image(d).to(device)
    x = x.unsqueeze(0) 
    h, w = x.size(2), x.size(3)
    pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
    x_padded = F.pad(x, pad, mode="constant", value=0)

    out_base = model.compress(x_padded, quality = 0)


    mu_base = out_base["mean_base"]
    std_base = out_base["std_base"]
    y_hat_base = out_base["y_hat"]
    y = out_base["y"]

    
    strings_z = out_base["strings"][1]
    strings_base = out_base["strings"][0]
    
    folder = os.path.join(savepath,nome_immagine)
    os.makedirs(folder,exist_ok = True)
    name =  os.path.join(folder,"base.pkl")
    with open(name, 'wb') as file:
        pickle.dump(strings_base, file)

    name =  os.path.join(folder,"z.pkl")
    with open(name, 'wb') as file:
        pickle.dump(strings_z, file)    
        





    bitstreams = {}

    bitstreams["shape"] = out_base["shape"]
    bitstreams["z"] = out_base["strings"][1]
    bitstreams["base"] = out_base["strings"][0]
    #bitstreams["top"] = []

    start_t = time.time()


    bitstreams_list, shape = extract_all_bitsreams(model,
                                        x_padded,
                                        y_hat_base,
                                        mu_base,
                                        std_base,
                                        q_list,
                                        rems = rems,
                                        y_checkpoint = None)

    end_t = time.time()
    bitstreams["top"] = bitstreams_list 
    bitstreams["y_shape"] = shape 
    bitstreams["unpad"] = unpad
    print("done: ",end_t-start_t) 
    return bitstreams 



def extract_all_bitsreams(model, x, y_hat_base, mu_base,std_base, q_list, y_checkpoint = None):


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
    y_checkpoint_hat = y_checkpoint.chunk(10,1) if y_checkpoint is not None else None #y_hat_base_slices  ###fff


    mu_base = mu_base.chunk(10,1)
    std_base = std_base.chunk(10,1)

 
    r_slices = []
    indexes_slices = []
    scale_slices = []

    for slice_index in range(model.ns0,model.ns1):

        current_index = slice_index%model.ns0
        y_slice = y_slices[slice_index]
        if model.delta_encode:
            r_slice = y_slice - y_slices[current_index]

        y_shape
        mu,mu_t, scale = extract_retrieve_entropy_parameters(current_index,
                                                            model,
                                                            mu_total,
                                                            std_total, 
                                                            y_hat_slices_base,
                                                            latent_means, 
                                                            latent_scales,
                                                            y_shape)

        mu_total.append(mu_t)
        std_total.append(scale)



        r_slice_quantize = model.gaussian_conditional.quantize(r_slice - mu,"symbols") #[1,nc,h,w]
        indexes_sl = model.gaussian_conditional.build_indexes(scale).int() #[1,ch,h,w]

        r_slice_quantize = r_slice_quantize.ravel().unsqueeze(0) #[1,1*ch*h*w] 
        indexes_sl = indexes_sl.ravel().unsqueeze(0) #[1,1*ch*h*w]
        scale_slice = scale.ravel().unsqueeze(0) #[1,1*ch*h*w]
        r_slices.append(r_slice_quantize)
        indexes_slices.append(indexes_sl)
        scale_slices.append(scale_slice)



    r_hat_slice = torch.cat(r_slices,dim = 0) #[10,ch*h*w]
    index_hat_slice = torch.cat(indexes_slices,dim = 0) #[10,ch*h*w]
    scale_hat_slice = torch.cat(scale_slices,dim = 0) # [10,ch*h*w]


    ordered_scale,std_ordering_index = torch.sort(scale_hat_slice, dim=1, descending=True)


    r_hat_slice_ordered = torch.gather(r_hat_slice,dim = 1, index= std_ordering_index) #[10,ch*h*w]
    ordered_index = torch.gather(index_hat_slice,dim = 1,index = std_ordering_index) #[10,ch*h*w]



    q_init = 0
    shapes =  r_hat_slice_ordered[0].shape[0]


    bitstream = []
    for j, qs in enumerate(q_list):
        print("----> ",qs)
        q_end = qs*10
        q_init = 0 if j == 0 else q_list[j-1]
        q_init = q_init*10
        init_length = int((q_init*shapes)/100) + 1 if j > 0 else 0 
        end_length =  int((q_end*shapes)/100) 

        r_hat_tbc = r_hat_slice_ordered[:,init_length:end_length + 1] #[10,init_l:end_l]
        ordered_index_bc = ordered_index[:,init_length:end_length + 1] #ddd

        symbols_list = torch.flatten(r_hat_tbc).unsqueeze(0) # [1,10*(end_length - init_length)]
        indexes_l = torch.flatten(ordered_index_bc).unsqueeze(0) # [1,10*K]
        time_s = time.time()
        symbols_q = model.gaussian_conditional.compress(symbols_list,
                                                             indexes_l,
                                                             already_quantize = True)
        time_e = time.time()
        print("time for encoding is ",time_e - time_s)
        bitstream.append(symbols_q)
    
    #ora inizia la parte di decoding 
    # ordered_scale,std_ordering_index = torch.sort(scale_hat_slice, dim=0, descending=True) 
    # ordered_index = torch.gather(index_hat_slice,dim = 0,index = std_ordering_index) #[10,ch*h*w]
    
    print("here we decode the stuff just to see if it works")
    print("TO DO:delete this part")
    for j in range(len(bitstream)):
        qs =q_list[j]

        q_end = qs*10
        q_init = 0 if j == 0 else q_list[j-1]
        q_init = q_init*10
        init_length = int((q_init*shapes)/100) + 1 if j > 0 else 0 
        end_length =  int((q_end*shapes)/100) 

        symbols = bitstream[j]
        ordered_index_bc = ordered_index[:,init_length:end_length + 1] #ddd
        indexes_l = torch.flatten(ordered_index_bc).unsqueeze(0)


        symbols_q = model.base_net.gaussian_conditional.decompress(symbols,
                                                             indexes_l,
                                                             already_quantize = True)
        

        r_hat_tbc = r_hat_tbc.reshape(1,10,-1)  #divido nelle sottosezioni
    
    return bitstream, shapes




