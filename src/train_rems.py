import wandb 
import torch 
import time 
import torch.optim as optim
from utility import ( parse_args, tri_planet_23_psnr, tri_planet_23_bpp,
                    tri_planet_22_bpp, tri_planet_22_psnr, plot_rate_distorsion, 
                    create_savepath, sec_to_hours, initialize_model_from_pretrained, configure_optimizers, save_checkpoint)
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ImageFolder, TestKodakDataset
from models import get_model
from training import RateLoss, compress_with_ac, train_one_epoch, valid_epoch, test_epoch

import os
import sys
import numpy as np




def main(argv):
    args = parse_args(argv)
    print(args)





    train_path =  args.training_dataset
    test_path = args.test_dataset
    save_path = args.save_path
    device = "cuda"


    if args.wandb_log:
        assert args.entity is not None
        wandb.init( config= args, project=args.project, entity=args.entity)
    


    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [ transforms.ToTensor()]
    )

    train_dataset = ImageFolder(train_path, split="train", transform=train_transforms, num_images=args.num_images)
    valid_dataset = ImageFolder(train_path, split="test", transform=train_transforms, num_images=args.num_images_val)
    test_dataset = TestKodakDataset(data_dir=test_path, transform= test_transforms)


    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.valid_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers,shuffle=False,pin_memory=(device == "cuda"),)




    filelist = test_dataset.image_path



    assert args.checkpoint != "none"


    checkpoint = torch.load(args.checkpoint, map_location="cuda")
    #print("vedo il checkpoint: ",checkpoint)
    checkpoint["args"].model = args.model
    args_save = checkpoint["args"]
    net = get_model(checkpoint["args"],device)
    net.load_state_dict(checkpoint["state_dict"]) #state_dict



    last_epoch = 0

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=args.patience)
    

    criterion = RateLoss()



    best_loss = float("inf")
    counter = 0
    epoch_enc = 0


    num_points_per_segments = args.num_points
    

    list_quality = []
    levels_extend = args.check_levels + [10]
    for i in range(len(levels_extend)-1):

        left = levels_extend[i] + 0.01 if i == 0 else levels_extend[i]
        right = levels_extend[i + 1] 
        delta = (right-left)/num_points_per_segments #ddd
        #print(i," ",delta)
        c = list(np.arange(left,right ,delta))
        c = [round(p,3) for p in c]
        list_quality.extend(c)



    if args.test_before:
        net.enable_rem = False
        pr_list = [0,0.05,0.1,0.25,0.5,0.6,0.75,1,1.25,2,3,5,10] 
        mask_pol = "point-based-std"

        bpp_init, psnr_init,_ = compress_with_ac(net, #net 
                                        filelist, 
                                        device,
                                        pr_list =pr_list,  
                                        mask_pol = mask_pol)
        
        net.enable_rem = True

        print("----> ",bpp_init," ",psnr_init)
    

    net.unfreeze_rems()


    print("************************************************************")
    print("********************** START TRAINING **********************")
    print("************************************************************")
    num_tainable = net.print_information()
    print("************************************************************")
    print("********************** START TRAINING **********************")
    print("************************************************************")    

    for epoch in range(last_epoch, args.epochs):
        print("******************************************************")
        print("epoch: ",epoch)
        start = time.time()
        #print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        num_tainable = net.print_information()
        if num_tainable > 0:
            counter ,loss_train, bpp_loss_train, mse_lss_train, bpp_scalable_train = train_one_epoch( net, 
                                      criterion, 
                                      train_dataloader, 
                                      optimizer, 
                                      aux_optimizer, 
                                      epoch, 
                                      counter,
                                      list_quality= list_quality,
                                      sampling_training = args.sampling_training,
                                      lmbda_list=None,
                                      wandb_log = args.wandb_log_train)
