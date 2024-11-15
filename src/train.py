import wandb 
import random 
import torch 
import time 
import math 
import torch.optim as optim
from utility import parse_args, tri_planet_23_psnr, tri_planet_23_bpp, tri_planet_22_bpp, tri_planet_22_psnr, plot_rate_distorsion, create_savepath, sec_to_hours
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ImageFolder, TestKodakDataset
from models import get_model
from training import ScalableRateDistortionLoss, RateDistortionLoss, DistortionLoss
from training import compress_with_ac, train_one_epoch, valid_epoch, test_epoch
import os
import sys




def save_checkpoint(state, is_best, last_pth,very_best):
    if is_best:
        torch.save(state, very_best)
        wandb.save(very_best)
    else:
        torch.save(state, last_pth)
        wandb.save(last_pth)



def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") 
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") 
    }

    
    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )

    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer



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




    net = get_model(args,device)
    net = net.to(device)
    #net.update()


    if args.checkpoint != "none":
        print("entro in checkpoint")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"],strict = True)

    net.update()



    last_epoch = 0

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=args.patience)
    
    if args.training_type == "first_train":
        criterion = ScalableRateDistortionLoss(lmbda_list=args.lmbda_list)
    elif args.training_type == "refine_gs":
        criterion = RateDistortionLoss()
    else:
        criterion = DistortionLoss()


    best_loss = float("inf")
    counter = 0
    epoch_enc = 0

    if args.training_type == "first_train":
        list_quality = [0,10]
        lmbda_list = None
    elif args.training_type == "refine_gs":
        list_quality = [0.025,0.05,0.1,0.25,0.5,0.75,0.76,1.0, 1.25,1.5,2,2.25,2.5,3,4,5,6,8,10]
        lmbda_list = None
    elif args.training_type == "refine_gs_ga":
        list_quality = [0.025,0.05,0.1,0.15,0.25,0.5,0.6,0.7,0.75,0.9,
                        1.0,1.15,1.25,1.5,1.75,
                        2,2.15,2.25,2.5,2.75,
                        3,3.5,
                        4,4.5,
                        5,6,8,10]
        start = torch.log10(torch.tensor(args.lmbda_list[0]))
        end = torch.log10(torch.tensor(args.lmbda_list[1]))
        lmbda_list = torch.logspace(start, end, steps=len(list_quality) + 1)[1:]
        
    else:
        raise NotImplementedError()

    if args.checkpoint != "none" and args.test_before:
        pr_list = [0,0.05,0.1,0.25,0.5,0.6,0.75,1,1.25,2,3,5,10] 
        mask_pol = "point-based-std"

        bpp_init, psnr_init,_ = compress_with_ac(net, #net 
                                        filelist, 
                                        device,
                                        pr_list =pr_list,  
                                        mask_pol = mask_pol)
    
        print("----> ",bpp_init," ",psnr_init) 

    #print("done everything")
    
    print("freeze part of th network according to ",args.training_type)
    if args.training_type ==  "refine_gs":
        net.freeze_all()
        net.unfreeze_decoder()
    elif args.training_type == "refine_gs_ga":
        net.freeze_all()
        net.unfreeze_decoder()
        net.unfreeze_encoder()  

    print("************************************************************")
    print("********************** START TRAINING **********************")
    print("************************************************************")
    num_tainable = net.print_information()
    print("************************************************************")
    print("********************** START TRAINING **********************")
    print("************************************************************")     
    return 0

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
                                      lmbda_list=lmbda_list)
            
        print("finito il train della epoca")

        if args.wandb_log:
            wand_dict = {
                "train": epoch,
                "train_batch/loss": loss_train,
                "train_batch/bpp_total": bpp_loss_train,
                "train_batch/mse":mse_lss_train,
                "train_batch/bpp_progressive":bpp_scalable_train,
            }
            wandb.log(wand_dict)
        
        
        loss = valid_epoch(epoch, 
                        valid_dataloader,
                        criterion, 
                        net,
                        pr_list = [0,10],
                        wandb_log = args.wandb_log,
                        lmbda_list=lmbda_list) 
        lr_scheduler.step(loss)
        print(f'Current patience level: {lr_scheduler.patience - lr_scheduler.num_bad_epochs}')

        list_pr = [0,0.25,0.5,1,1.5,2,2.5,3,3.5,4,5,6,10]
        mask_pol = "point-based-std"
        bpp_t, psnr_t = test_epoch(epoch, 
                       test_dataloader,
                       net, 
                       pr_list = list_pr,
                       wandb_log = args.wandb_log
                       )
        print("finito il test della epoca: ",bpp_t," ",psnr_t)



        end = time.time()
        print("Runtime of the epoch:  ", epoch)
        sec_to_hours(end - start) 
        print("END OF EPOCH ", epoch)


        is_best = loss < best_loss
        best_loss = min(loss, best_loss)



        if epoch%5==0 or is_best:
            net.update()
            #net.lmbda_list
            bpp, psnr,_ = compress_with_ac(net,  
                                            filelist, 
                                            device,
                                           epoch = -10, 
                                           pr_list =list_pr,  
                                            mask_pol = mask_pol,
                                            writing = None)


            psnr_res = {}
            bpp_res = {}

            bpp_res["our"] = bpp
            psnr_res["our"] = psnr

            psnr_res["base"] =   [29.20, 30.59,32.26,34.15,35.91,37.72]
            bpp_res["base"] =  [0.127,0.199,0.309,0.449,0.649,0.895]

            bpp_res["tri_planet_23"] = tri_planet_23_bpp
            psnr_res["tri_planet_23"] = tri_planet_23_psnr

            bpp_res["tri_planet_22"] = tri_planet_22_bpp
            psnr_res["tri_planet_22"] = tri_planet_22_psnr

            if args.wandb_log:
                plot_rate_distorsion(bpp_res, psnr_res,epoch_enc, eest="compression")
                bpp_res["our"] = bpp_t
                psnr_res["our"] = psnr_t          
                plot_rate_distorsion(bpp_res, psnr_res,epoch_enc, eest="model")
                epoch_enc += 1



            cartella = os.path.join(save_path,args.code) #dddd
            os.makedirs(cartella,exist_ok=True) 


            last_pth, very_best =  create_savepath( cartella)



            #if is_best is True or epoch%10==0 or epoch > 98: #args.save:
            save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "aux_optimizer":aux_optimizer.state_dict() if aux_optimizer is not None else "None",
                        "args":args
        
                    },
                    is_best,
                    last_pth,
                    very_best
                    )

            log_dict = {
            "train":epoch,
            "train/leaning_rate": optimizer.param_groups[0]['lr']
            #"train/beta": annealing_strategy_gaussian.bet
            }
            wandb.log(log_dict)


    print("training completed")





if __name__ == "__main__":
    #Enhanced-imagecompression-adapter-sketch
    main(sys.argv[1:])



