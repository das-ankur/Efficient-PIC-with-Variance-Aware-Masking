import torch 
import wandb
from test import parse_args_demo, read_and_pads_image, encode,  decode
import time 
from models import get_model
from utility import sec_to_hours, compute_psnr
import torch.nn.functional as F 
import sys


def main(argv):

    q_levs = [0.02,0.05,0.5,0.75,1,1.5,2,2.5,3,4,5,5.5,6,6.6,10] 
    args = parse_args_demo(argv)
    print(args)
    if args.wandb:
        wandb.init( config= args, project=args.project, entity=args.entity)  
    
    
    device = args.device

    print("Initializing and loading the model")
    start_t = time.time()
    checkpoint = torch.load(args.checkpoint, map_location="cuda")
    checkpoint["args"].model = args.model
    net = get_model(checkpoint["args"],device)
    net.load_state_dict(checkpoint["state_dict"], strict = True) #state_dict

    print("initialization is over.")
    print("time for initialization is")
    sec_to_hours(time.time() - start_t) #ddd


    path_save = args.path_save
    path_image = args.path_image
    name_image = path_image.split("/")[-1].split(".")[0]
    x, x_padded, unpad = read_and_pads_image(path_image,device)

    ql = [0] + q_levs
    for i,c in enumerate(ql):
        ql[i] = c*10

    print("start encoding following this q_list: ",ql)
    start_enc = time.time()
    bitstreams = encode(net, x_padded, path_save, name_image , q_list = q_levs,rems = args.rems)
    end_enc = time.time()
    print("time for encoding")
    sec_to_hours(end_enc - start_enc)


    bitstreams["unpad"] = unpad


    print("decode level base")
    shape = bitstreams["y_shape"]
    start_dec_base = time.time()
    rec_hat_base  = decode(net, bitstreams, shape, q_ind = 0)
    end_dec_base = time.time()
    print("time for decoding first base level")
    sec_to_hours(end_dec_base - start_dec_base)

    y_hat_base = rec_hat_base["y_hat"]



    for qk in args.requested_levels:
        print("decoding qk=====> ",qk)
        start_dec_time = time.time()
        recs  = decode(net, bitstreams, shape, q_ind = qk, y_hat_base=y_hat_base)
        end_dec_time = time.time()
        print("time for decoding: ",sec_to_hours(end_dec_time - start_dec_time))
        recs["x_hat"] = F.pad(recs["x_hat"], unpad)
        recs["x_hat"].clamp_(0.,1.)  
        psnr_im = compute_psnr(x, recs["x_hat"])
        print("the psnr is ",psnr_im)

        

        
    


    
if __name__ == "__main__":
    #Enhanced-imagecompression-adapter-sketch
    main(sys.argv[1:])











