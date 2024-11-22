import argparse
import os
import time



def parse_args_demo(argv):
    parser = argparse.ArgumentParser(description="Example training script.") #dddddddd
    parser.add_argument("--path",type=str,default = "/scratch/ResDSIC/models/")
    parser.add_argument("--data_path",type=str,default = "/scratch/dataset/kodak/kodim01.png")
    parser.add_argument("--model",type=str,default = "pic")
    parser.add_argument("--device",type=str,choices=["cpu","cuda"],default = "cuda")
    parser.add_argument("--rems", action="store_true", help="Use cuda") #/scratch/ScalableResults
    parser.add_argument("--path_save",type=str,default = "/scratch/ScalableResults/bits")
    parser.add_argument("--path_image",type=str,default = "/scratch/dataset/kodak/kodim12.png")


    parser.add_argument("--seed", type=float,default = 42, help="Set random seed for reproducibility")
    args = parser.parse_args(argv) #dddd
    return args