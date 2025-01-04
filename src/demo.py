import torch 
from test import parse_args_demo, read_and_pads_image, encode,  decode
import time 
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import TestKodakDataset
from models import get_model
from utility import sec_to_hours, compute_psnr
import torch.nn.functional as F 
import sys
from training import ScalableRateDistortionLoss, RateDistortionLoss, DistortionLoss, RateLoss
from training import compress_with_ac,test_epoch

def main(argv):

    
    args = parse_args_demo(argv)
    q_levs = args.q_levs
    print(args)

    
    
    device = args.device

    print("Initializing and loading the model")
    start_t = time.time()
    checkpoint = torch.load(args.checkpoint, map_location="cuda")
    checkpoint["args"].model = args.model
    net = get_model(checkpoint["args"],device)
    net.load_state_dict(checkpoint["state_dict"], strict = True) #state_dict

    print("initialization is over.")
    print("time for initialization is ",sec_to_hours(time.time() - start_t,rt = True))



    save_path = args.save_path
    path_image = args.path_image



    if args.fast_encdec:

        print("perform different encoding and decoding for each quality. Faster solution with same results")
        pr_list = [0] + q_levs 
        mask_pol = "point-based-std"
        rems = None if args.rems is False else net.check_levels
        filelist = [os.path.join("/scratch/dataset/kodak",f) for f in os.listdir("/scratch/dataset/kodak")]#[path_image]
        bpp_image, psnr_image,_ = compress_with_ac(net, #net 
                                    filelist, 
                                    device,
                                    pr_list =pr_list,
                                    rems = rems,  
                                    mask_pol = mask_pol)
        print("results for image: ", path_image.split("/")[-1].split(".")[0])
        for i in range(len(bpp_image)):
            print("quality ",pr_list[i]*10,": bpp = ",bpp_image[i]," psnr = ",psnr_image[i])
        print("done")  

    else:   
        if args.requested_levels is None: 
            args.requested_levels = np.arange(1,len(q_levs))
        else:
            b = c[0]
            for c in args.requested_levels:
                assert c < len(q_levs) 
                assert c >= b 
                b = c
        
                
        x, x_padded, unpad = read_and_pads_image(path_image,device)
        num_pixels = x.shape[2]*x.shape[3]


        ql = [0] + q_levs
        for i,c in enumerate(ql):
            ql[i] = c*10

        print("start encoding following this q_list: ",ql) #ddd
        start_enc = time.time()
        with torch.no_grad():

            bitstreams,bits = encode(net, x_padded, save_path = save_path , q_list = q_levs,rems = args.rems)
        end_enc = time.time()
        
        print("time for encoding: ",sec_to_hours(end_enc - start_enc,rt = True))
        
        

        bpp_base = bits[1]/num_pixels
        bpp_hype = bits[0]/num_pixels
  
        print("decode level base")
 
        start_dec_base = time.time()
        rec_hat_base  = decode(net, bitstreams, q_ind = 0)
        end_dec_base = time.time()

        c = rec_hat_base["x_hat"]
        c = F.pad(c, unpad)
        c.clamp_(0.,1.)   
        psnr_im = compute_psnr(x, c)
        print("Base level, psnr = ",round(psnr_im,3),", bpp = ",round(bpp_base + bpp_hype,3)," time: ",sec_to_hours(end_dec_base - start_dec_base,rt = True))
        


        y_hat_base = rec_hat_base["y_hat"]
        z_data = None 
        entropy_data = None
        for jj,qk in enumerate(args.requested_levels[1:]):
            start_dec_time = time.time()
            with torch.no_grad():
                recs  = decode(net, 
                               bitstreams, 
                               q_ind = qk, 
                               y_hat_base = y_hat_base,
                               z_data = z_data, 
                               entropy_data= entropy_data)
            

            end_dec_time = time.time()
            z_data = recs["z_data"]
            entropy_data = recs["entropy_data"]
            recs["x_hat"] = F.pad(recs["x_hat"], unpad)
            recs["x_hat"].clamp_(0.,1.)  
            psnr_im = compute_psnr(x, recs["x_hat"])
            progressive_bpp = sum(bits[2][:jj])/num_pixels 
            print("Level ",q_levs[qk],": psnr = ",round(psnr_im,3)," bpp = ",round(progressive_bpp + bpp_base + bpp_hype,3)," time: ",sec_to_hours(end_dec_time - start_dec_time,rt = True))

            
            

        

        
    


    
if __name__ == "__main__":
    #Enhanced-imagecompression-adapter-sketch
    main(sys.argv[1:])











