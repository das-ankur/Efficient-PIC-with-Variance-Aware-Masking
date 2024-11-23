

from utility import  read_image, compute_padding
import torch.nn.functional as F 
import os
import pickle
import time
import torch
from .utils import extract_retrieve_entropy_parameters 
from utility import sec_to_hours

q_list = [0.002,0.05,0.5,0.75,1,1.5,2,2.5,3,4,5,5.5,6,6.6,10] 

def encode(model, x_padded, path_save, name_image,rems = False,q_list = q_list):


    print("encode level base")
    start_base = time.time()
    out_base = model.compress(x_padded, quality = 0)
    end_base = time.time()
    print("Done encoding base: ",end_base - start_base)


    mu_base = out_base["mean_base"]
    std_base = out_base["scale_base"]
    y_hat_base = out_base["y_hat_base"]
    #y = out_base["y"]

    
    strings_z = out_base["strings"][1]
    strings_base = out_base["strings"][0]
    
    folder = os.path.join(path_save,name_image)
    os.makedirs(folder,exist_ok = True)
    name =  os.path.join(folder,"base.pkl")
    with open(name, 'wb') as file:
        pickle.dump(strings_base, file)

    name =  os.path.join(folder,"z.pkl")
    with open(name, 'wb') as file:
        pickle.dump(strings_z, file)    
        





    bitstreams = {}
    bitstreams["q_list"] = q_list
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
    #bitstreams["unpad"] = unpad
    print("Done encoding progressive parts: ",end_t-start_t) 
    return bitstreams 



def extract_all_bitsreams(model, x, y_hat_base, mu_base,std_base, q_list, y_checkpoint = None, rems = False):


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
    #print("one cycle for")
    #start_t = time.time()
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

        mu_total.append(mu_t)
        std_total.append(scale)


        r_slice_zero_mu = r_slice - mu

        r_slice_quantize = model.gaussian_conditional.quantize(r_slice_zero_mu,"symbols") #[1,nc,h,w]
        indexes_sl = model.gaussian_conditional.build_indexes(scale).int() #[1,ch,h,w]

        r_slice_quantize = r_slice_quantize.ravel().unsqueeze(0) #[1,1*ch*h*w] 
        indexes_sl = indexes_sl.ravel().unsqueeze(0) #[1,1*ch*h*w]
        scale_slice = scale.ravel().unsqueeze(0) #[1,1*ch*h*w]
        r_slices.append(r_slice_quantize)
        indexes_slices.append(indexes_sl)
        scale_slices.append(scale_slice)
    #end_t = time.time()
    #print("end for cycle: ",end_t - start_t)


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
        #print("start encoding level ",qs)
        start_qs = time.time()
        q_end = qs*10
        q_init = 0 if j == 0 else q_list[j-1]
        #q_init = q_list[j-1]
        q_init = q_init*10
        init_length = int((q_init*shapes)/100) + 1 if j > 0 else 0 
        end_length =  int((q_end*shapes)/100) 


        r_hat_tbc = r_hat_slice_ordered[:,init_length:end_length + 1] #[10,init_l:end_l]
        ordered_index_bc = ordered_index[:,init_length:end_length + 1] #ddd



        symbols_list = torch.flatten(r_hat_tbc).unsqueeze(0) # [1,10*(end_length - init_length)]
        indexes_l = torch.flatten(ordered_index_bc).unsqueeze(0) # [1,10*K]


        symbols_q = model.gaussian_conditional.compress(symbols_list,
                                                             indexes_l,
                                                             already_quantize = True)

        prova =     model.gaussian_conditional.decompress(symbols_q,
                                                             indexes_l,
                                                             )
        
        if j == 0:
            print("prova a vedere ",torch.equal(prova,symbols_list))

        end_qs = time.time()
        #print("time for encoding is ",end_qs - start_qs)
        bitstream.append(symbols_q)
    

    
    for j in range(len(bitstream)):
        qs =q_list[j]
        symbols = bitstream[j]

        q_end = qs*10
        q_init = 0 if j == 0 else q_list[j-1]
        q_init = q_init*10
        init_length = int((q_init*shapes)/100) + 1 if j > 0 else 0 
        end_length =  int((q_end*shapes)/100) 

        
        ordered_index_bc = ordered_index[:,init_length:end_length + 1] #ddd
        indexes_l = torch.flatten(ordered_index_bc).unsqueeze(0)



        symbols_q = model.gaussian_conditional.decompress(symbols,
                                                             indexes_l)


        r_hat_tbc = symbols_q.float()#reshape(10,-1).unsqueeze(0).float()  #divido nelle sottosezioni
        
        #print("dec ",j," shape is ",r_hat_tbc.shape)



    #print("check if it is the same (hope) ",r_dec[0].shape)
    #print(torch.unique(r_dec[0],return_counts =True))
    


    return bitstream, shapes




