

import argparse
from models import models
def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.") #dddd

    parser.add_argument("--all_scalable", action="store_true", help="implement full scalable model")
    parser.add_argument("--aux-learning-rate", default=1e-3, type=float, help="Auxiliary loss learning rate (default: %(default)s)",)

    parser.add_argument( "--batch_size", type=int, default=16, help="batch_size")

    parser.add_argument("--clip_max_norm",default=1.0,type=float,help="gradient clipping max norm (default: %(default)s",)
    parser.add_argument("--code", type=str, default = "0001", help="Batch size (default: %(default)s)")
    parser.add_argument("--checkpoint", type=str, default = "/scratch/ResDSIC/demo/l2_train/2l_memdmh_mem5/_very_best.pth.tar") #dddd
    parser.add_argument("--cuda", action="store_true", help="Use cuda")

    parser.add_argument("--division_dimension", nargs='+', type=int, default = [320, 640])
    parser.add_argument( "--dim_chunk", type=int, default=32, help="dim chunk")
    parser.add_argument("--delta_encode", action="store_true", help="delta encoder")

    parser.add_argument("-e","--epochs",default=150,type=int,help="Number of epochs (default: %(default)s)",)
    parser.add_argument("--entity", type=str, default = "alberto-presta", help="entity name for wandb")

    parser.add_argument( "-lr", "--learning-rate", default=1e-4, type=float, help="Learning rate (default: %(default)s)",)
    parser.add_argument("--lmbda_list", nargs='+', type=float, default = [ 0.0055,0.04])
    parser.add_argument("--list_quality", nargs='+', type=int, default = [0,10])
    
    parser.add_argument("--mask_policy", type=str, default = "point-based-std", help="mask_policy")
    parser.add_argument("--multiple_decoder", action="store_true", help="Use cuda")
    parser.add_argument("--multiple_encoder", action="store_true", help="Use cuda")
    parser.add_argument("--multiple_hyperprior", action="store_true", help="Use cuda")
    parser.add_argument("-m","--model",default="pic",choices=models.keys(),help="Model architecture (default: %(default)s)",)    
    parser.add_argument("--M", type=int, default=640, help="M")


    parser.add_argument("-n","--num-workers",type=int,default=8,help="Dataloaders threads (default: %(default)s)",) #u_net_post
    parser.add_argument("--num_images", type=int, default=300000, help="num images") #ddddddd
    parser.add_argument("--num_images_val", type=int, default=816, help="Batch size (default: %(default)s)")
    parser.add_argument("--N", type=int, default=192, help="N")#ddddd#ddd


    parser.add_argument("--patch-size",type=int,nargs=2,default=(256, 256),help="Size of the patches to be cropped (default: %(default)s)",)
    parser.add_argument("--patience", type=int, default=7, help="patience")
    parser.add_argument("--project", type=str, default = "PIC", help="project name for wandb")
    


    parser.add_argument("--save_path",  type=str, default = "/scratch/WACV/models", help="test path")
    parser.add_argument("--save_images", type=str, default = "none", help = "save compressed images") #dddd
    parser.add_argument("--seed", type=float, help="Set random seed for reproducibility")
    parser.add_argument("--sampling_training", action="store_true", help="Save model to disk")
    parser.add_argument("--support_progressive_slices",default=4,type=int,help="support_progressive_slices",) #ssss

    parser.add_argument("--total_mu_rep", action="store_true", help="use entire mu as input")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Test batch size (default: %(default)s)", )
    parser.add_argument("--test_before", action="store_true", help="test before start training")
    parser.add_argument("--training_dataset",  type=str, default = "/scratch/dataset/openimages", help="project name for wandb")
    parser.add_argument("--test_dataset",  type=str, default = "/scratch/dataset/kodak", help="test path")
    parser.add_argument("--training_type",  type=str, default = "first_train", help="test path")
    #training_type

    parser.add_argument("--valid_batch_size",type=int,default=16,help="Test batch size (default: %(default)s)",)


    parser.add_argument("--wandb_log", action="store_true", help="Use cuda")
    parser.add_argument("--writing", type=str, default = "none", help = "writing test results on file txt") #dddd
    

    args = parser.parse_args(argv)
    return args