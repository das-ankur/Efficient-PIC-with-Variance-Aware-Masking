import argparse
import os
import time



def parse_args_demo(argv):
    parser = argparse.ArgumentParser(description="Example training script.") #dddddddd


    parser.add_argument("--checkpoint", type=str, default = "/scratch/ResDSIC/demo/l2_train/2l_memdmh_mem5/_very_best.pth.tar") #"/scratch/ResDSIC/demo/l2_train/2l_memdmh_mem5/_very_best.pth.tar"
    parser.add_argument("--path",type=str,default = "/scratch/ResDSIC/models/")
    parser.add_argument("--data_path",type=str,default = "/scratch/dataset/kodak/kodim01.png")
    parser.add_argument("--model",type=str,default = "pic")
    parser.add_argument("--device",type=str,choices=["cpu","cuda"],default = "cuda")
    parser.add_argument("--rems", action="store_true", help="Use cuda") #/scratch/ScalableResults
    parser.add_argument("--path_save",type=str,default = "/scratch/ScalableResults/bits")
    parser.add_argument("--path_image",type=str,default = "/scratch/dataset/kodak/kodim12.png")

    #requested_levels
    parser.add_argument("--requested_levels", nargs='+', type=int, default = [0,1,2,3])
    parser.add_argument("--seed", type=float,default = 42, help="Set random seed for reproducibility")
    parser.add_argument("--wandb", action="store_true", help="Use cuda")
    args = parser.parse_args(argv) #dddd
    return args