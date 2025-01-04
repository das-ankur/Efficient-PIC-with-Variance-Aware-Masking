import argparse
import os
import time



def parse_args_demo(argv):
    parser = argparse.ArgumentParser(description="Example training script.") #dddddddd


    parser.add_argument("--checkpoint", type=str, default = "/scratch/WACV/models/NoRems/_last.pth.tar") #"/scratch/ResDSIC/demo/l2_train/2l_memdmh_mem5/_very_best.pth.tar"
    parser.add_argument("--model",type=str,default = "pic")
    parser.add_argument("--device",type=str,choices=["cpu","cuda"],default = "cuda")
    parser.add_argument("--rems", action="store_true", help="Use cuda") #/scratch/ScalableResults
    parser.add_argument("--fast_encdec", action="store_true", help="Use cuda")
    
    parser.add_argument("--path_image",type=str,default = "/scratch/dataset/kodak/kodim12.png")

    #requested_levels
    parser.add_argument("--q_levs", nargs='+', type=float, default = [0.01,0.05,0.1,0.25,0.5,0.6,0.7,0.8,0.9,1,2,3,4,4.5,10])
    parser.add_argument("--requested_levels", nargs='+', type=int, default = None)
    parser.add_argument("--save_path",type=str,default = None)
    parser.add_argument("--seed", type=float,default = 42, help="Set random seed for reproducibility")
    args = parser.parse_args(argv) #dddd
    return args