import wandb 
import torch 
import time 
import torch.optim as optim
from utility import ( parse_args, tri_planet_23_psnr, tri_planet_23_bpp, psnr_best, bpp_best,
                    tri_planet_22_bpp, tri_planet_22_psnr, plot_rate_distorsion, 
                    create_savepath, sec_to_hours, initialize_model_from_pretrained, configure_optimizers, save_checkpoint)
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ImageFolder, TestKodakDataset
from models import get_model
from training import ScalableRateDistortionLoss, RateDistortionLoss, DistortionLoss, RateLoss
from training import compress_with_ac, train_one_epoch, valid_epoch, test_epoch
import os
import sys
import numpy as np #ciaos





def main(argv):
    args = parse_args(argv)
    print(args)





    train_path =  args.training_dataset
    test_path = args.test_dataset
    save_path = args.save_path
    device = "cuda"


    if args.wandb_log:
        assert args.entity is not None
        if args.training_type == "first_train":
            wandb.init( config= args, project="PIC-SCRATCH", entity=args.entity)
        else:
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




    #net = get_model(args,device)
    #net = net.to(device)
    #net.update()


    if args.checkpoint != "none":
        print("entro in checkpoint")
        checkpoint = torch.load(args.checkpoint, map_location="cuda")
        #print("vedo il checkpoint: ",checkpoint)
        

        
        if args.model == "rem":
            print("entro qua!")
            checkpoint["args"].check_levels =  args.check_levels
            checkpoint["args"].mu_std =  args.mu_std
            checkpoint["args"].dimension =  args.dimension    
            
        print("qua è il nuovo args con cui imposto il modello!! ", checkpoint["args"]) 
                   
        
        checkpoint["args"].model = args.model
        args_save = checkpoint["args"]
        net = get_model(checkpoint["args"],device)
        net.load_state_dict(checkpoint["state_dict"], strict = True) #state_dict
    else:
        if args.checkpoint_base != "none":
            net = get_model(args,device)
            base_checkpoint = torch.load(args.checkpoint_base,map_location=device)
            new_check = initialize_model_from_pretrained(base_checkpoint, args)
            net.load_state_dict(new_check,strict = False)
            args_save = args
        else:
            net = get_model(args,device)
            args_save = args
    net = net.to(device)
    net.update()



    last_epoch = 0

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=args.patience)
    
    if args.training_type == "first_train":
        criterion = ScalableRateDistortionLoss(lmbda_list=args.lmbda_list)
    elif args.training_type == "refine_gs_ga":
        criterion = RateDistortionLoss()
    elif args.training_type == "rems":
        criterion = RateLoss()
    else:
        criterion = DistortionLoss()


    best_loss = float("inf")
    counter = 0
    epoch_enc = 0

    if args.training_type == "first_train":
        list_quality = [0,10]
        lmbda_list = None
        rems = None
    elif args.training_type == "refine_gs":

        print("entro qua per il list_pr")
        list_pr_1 = list(np.arange(0.015,1.5 ,(1.5 - 0.025)/200)) + [1.5]
        list_pr_2 = list(np.arange(1.6,10 ,(10 - 1.6)/50)) + [10]
        list_quality = list_pr_1 + list_pr_2
        lmbda_list = None
        rems = args.check_levels
    elif args.training_type == "refine_gs_ga":
        list_pr_1 = list(np.arange(0.015,1.5 ,(1.5 - 0.025)/200)) + [1.5]
        list_pr_2 = list(np.arange(1.6,10 ,(10 - 1.6)/50)) + [10]
        list_quality = list_pr_1 + list_pr_2
        start = torch.log10(torch.tensor(args.lmbda_list[0]))
        end = torch.log10(torch.tensor(args.lmbda_list[1]))
        lmbda_list = torch.logspace(start, end, steps=len(list_quality) + 1)[1:]
        rems = None
    
    elif args.training_type == "rems":
        list_quality = []
        check_levels = args.check_levels + [10]
        for i in range(len(check_levels) - 1):
            sequent = check_levels[i + 1]
            current = check_levels[i]
            if i == 0:
                list_pr_1 = list(np.arange(current + 0.01,sequent ,(sequent - current)/args.check_levels_np[i])) 
            else:
                list_pr_1 = list(np.arange(current,sequent ,(sequent - current)/args.check_levels_np[i])) 
            list_quality.extend(list_pr_1) 
        list_quality = [round(x, 4) for x in list_quality] 
        if 10 not in list_quality:
            list_quality.extend([10])
        lmbda_list = None
        rems = args.check_levels
    else:
        raise NotImplementedError()

    if args.checkpoint != "none" and args.test_before:
        pr_list = [0,0.05,0.1,0.25,0.5,0.6,0.75,1,1.25,2,3,5,10] 
        mask_pol = "point-based-std"
        if args.model == "rem":
            net.enable_rem = [False for i in range(len(net.check_levels))]
        bpp_init, psnr_init,_ = compress_with_ac(net, #net 
                                        filelist, 
                                        device,
                                        pr_list =pr_list,  
                                        mask_pol = mask_pol)
        net.enable_rem = [True for i in range(len(net.check_levels))]
        print("----> ",bpp_init," ",psnr_init) 
    

    #print("done everything")

    print("freeze part of th network according to ",args.training_type)
    if args.training_type ==  "refine_gs":
        net.freeze_all()
        net.unfreeze_decoder(lrp = args.lrp)
    elif args.training_type == "refine_gs_ga":
        net.freeze_all()
        net.unfreeze_decoder()
        net.unfreeze_encoder()  
    elif args.training_type == "rems":
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
                                      lmbda_list=lmbda_list,
                                      rems = rems,
                                      wandb_log = args.wandb_log_train)
            
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
                        rems = rems,
                        wandb_log = args.wandb_log,
                        lmbda_list=lmbda_list) 
        lr_scheduler.step(loss)
        print(f'Current patience level: {lr_scheduler.patience - lr_scheduler.num_bad_epochs}')

        list_pr = [0,0.05,0.1,0.25,0.5,0.6,0.75,1,1.25,2,3,5,10] 
        mask_pol = "point-based-std"
        bpp_t, psnr_t = test_epoch( 
                       test_dataloader = test_dataloader,
                       model = net, 
                       criterion = criterion,
                       pr_list = list_pr,
                       wandb_log = args.wandb_log,
                       rems = rems
                       )
        print("finito il test della epoca: ",bpp_t," ",psnr_t)



        end = time.time()
        print("Runtime of the epoch:  ", epoch)
        sec_to_hours(end - start) 
        print("END OF EPOCH ", epoch)


        is_best = loss < best_loss
        best_loss = min(loss, best_loss)



        if epoch%2==0 or is_best:
            net.update()
            #net.lmbda_list
            bpp, psnr,_ = compress_with_ac(net,  
                                            filelist, 
                                            device,
                                          
                                           pr_list =list_pr,  
                                            mask_pol = mask_pol
                                           )


            psnr_res = {}
            bpp_res = {}
            print("la compression come è andata: ",bpp, "  ",psnr)
            bpp_res["our"] = bpp
            psnr_res["our"] = psnr


            if args.test_before:
                bpp_res["base"] =   bpp_init
                psnr_res["base"] =  psnr_init
            else:
                psnr_res["base"] =   [29.20, 30.59,32.26,34.15,35.91,37.72]
                bpp_res["base"] =  [0.127,0.199,0.309,0.449,0.649,0.895]


                bpp_res["best"] = bpp_best
                psnr_res["best"] =  psnr_best

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
                        "args":args_save
        
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



