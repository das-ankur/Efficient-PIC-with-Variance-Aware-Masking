from utility import AverageMeter, read_image, compute_msssim, compute_psnr, compute_padding
import random
import torch
import wandb 
import random 
import torch.nn.functional as F 
import torchvision.transforms as transforms
import time
import math 
from torch.nn.functional import mse_loss

def train_one_epoch(model, 
                    criterion, 
                    train_dataloader,
                      optimizer, 
                      aux_optimizer, 
                      epoch, 
                      counter,
                      sampling_training = False,
                      list_quality = None,
                      clip_max_norm = 1.0,
                      wandb_log = False):
    model.train()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_lss = AverageMeter()
    bpp_scalable = AverageMeter()

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        if aux_optimizer is not None:
            aux_optimizer.zero_grad()

        if sampling_training:
            quality_index =  random.randint(0, len(list_quality) - 1)
            quality = list_quality[quality_index]
            out_net = model.forward_single_quality(d, quality = quality, training = False)
            out_criterion = criterion(out_net, d)
        else:
            out_net = model(d, quality = list_quality)
            out_criterion = criterion(out_net, d)
        
        out_criterion["loss"].backward()

        if aux_optimizer is not None:
            aux_loss = model.aux_loss() 
            aux_loss.backward()
            aux_optimizer.step()


        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()


        loss.update(out_criterion["loss"].clone().detach())
        mse_lss.update(out_criterion["mse_loss"].mean().clone().detach())
        bpp_loss.update(out_criterion["bpp_loss"].clone().detach())
        bpp_scalable.update(out_criterion["bpp_scalable"].clone().detach())

        if wandb_log:
            wand_dict = {
                "train_batch": counter,
                "train_batch/loss": out_criterion["loss"].clone().detach().item(),
                "train_batch/bpp_total": out_criterion["bpp_loss"].clone().detach().item(),
                "train_batch/mse":out_criterion["mse_loss"].mean().clone().detach().item(),
                "train_batch/bpp_progressive":out_criterion["bpp_scalable"].clone().detach().item(),
            }
            wandb.log(wand_dict)
        
        counter += 1

        if i % 1000 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].mean().item() * 255 ** 2 / 3:.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
               f"\tAux loss: {0.000:.2f}"
            )

    return counter, loss.avg, bpp_loss.avg, mse_lss.avg, bpp_scalable.avg 





def valid_epoch(epoch, test_dataloader,criterion, model, pr_list = [0.05], wandb_log = True):
    #pr_list =  [0] +  pr_list  + [-1]
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter() 
    bpp_loss =AverageMeter() 
    mse_lss = AverageMeter() 
    
    psnr = AverageMeter() 
    mutual_info = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:

            d = d.to(device)
            for _,p in enumerate(pr_list):

                out_net = model.forward_single_quality(d, quality = p, training = False, mask_pol = "point-based-std")

                psnr_im = compute_psnr(d, out_net["x_hat"])
                batch_size_images, _, H, W =d.size()
                num_pixels = batch_size_images * H * W
                denominator = -math.log(2) * num_pixels
                likelihoods = out_net["likelihoods"]
                bpp = (torch.log(likelihoods["y"]).sum() + torch.log(likelihoods["z"]).sum())/denominator

                    
                bpp_loss.update(bpp)

                mse_lss.update(mse_loss(d, out_net["x_hat"]))
                psnr.update(psnr_im) 

                        #out_net = model(d)
                out_criterion = criterion(out_net, d) #dddddd

                loss.update(out_criterion["loss"].clone().detach())

         

    if wandb_log is False: 
        log_dict = {
                "valid":epoch,
                "valid/loss": loss.avg,
                "valid/bpp":bpp_loss.avg,
                "valid/mse": mse_lss.avg,
                "valid/psnr":psnr.avg,

            #   "test/y_loss_"+ name[i]: y_loss[i].avg,
                }
        wandb.log(log_dict)

    return loss.avg





def test_epoch(epoch, test_dataloader, model, pr_list, wandb_log = False):
    model.eval()
    device = next(model.parameters()).device


    bpp_loss =[AverageMeter()  for _ in range(len(pr_list))] 
    psnr = [AverageMeter()  for _ in range(len(pr_list))]
    mutual_info = [AverageMeter()  for _ in range(len(pr_list))]

    with torch.no_grad():
        for d,_ in test_dataloader:
            d = d.to(device)
            for j,p in enumerate(pr_list):
                out_net = model.forward_single_quality(d, quality = p, training = False,  mask_pol = "point-based-std")

                psnr_im = compute_psnr(d, out_net["x_hat"])
                batch_size_images, _, H, W =d.size()
                num_pixels = batch_size_images * H * W
                denominator = -math.log(2) * num_pixels #dddd
                likelihoods = out_net["likelihoods"]
                bpp = (torch.log(likelihoods["y"]).sum() + torch.log(likelihoods["z"]).sum())/denominator


                psnr[j].update(psnr_im)
                bpp_loss[j].update(bpp)


    if wandb_log:
        for i in range(len(pr_list)):
            if i== 0:
                name = "test_base"
            elif i == len(pr_list) - 1:
                name = "test_complete"
            else:
                c = str(pr_list[i])
                name = "test_quality_" + c 
            
            log_dict = {
                name:epoch,
                name + "/bpp":bpp_loss[i].avg,
                name + "/psnr":psnr[i].avg,
                }

            wandb.log(log_dict)

            if "mutual" in list(out_net):
                log_dict = {
                    name:epoch,
                    name + "/mutual":mutual_info[i].avg
                    }
                wandb.log(log_dict)

    return [bpp_loss[i].avg for i in range(len(bpp_loss))], [psnr[i].avg for i in range(len(psnr))]



#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
def compress_with_ac(model,  
                    filelist, 
                    device, 
                    pr_list = [0.05,0.01], 
                    mask_pol = None, 
                    writing = None, 
                    save_images = False):

    l = len(pr_list)
    print("ho finito l'update")
    bpp_loss = [AverageMeter() for _ in range(l)]
    psnr =[AverageMeter() for _ in range(l)]
    mssim = [AverageMeter() for _ in range(l)]
    dec_time = [AverageMeter() for _ in range(l)]



    for i,d in enumerate(filelist):
        print("l'immagine d: ",d," ",i)
        custom_map = None 

        with torch.no_grad():
            x = read_image(d).to(device)
            nome_immagine = d.split("/")[-1].split(".")[0]
            x = x.unsqueeze(0) 
            h, w = x.size(2), x.size(3)
            pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
            x_padded = F.pad(x, pad, mode="constant", value=0)
        

            for j,p in enumerate(pr_list):
                
                lev = "level_" + str(j)

                

                data =  model.compress(x_padded, quality =p, mask_pol = mask_pol,)
                #if j%8==0:
                #    print(sum(len(s[0]) for s in data["strings"][0]))
                start = time.time()
                out_dec = model.decompress(data["strings"], data["shape"], quality = p,mask_pol = mask_pol)
                end = time.time()
                #print("Runtime of the epoch:  ", epoch)
                decoded_time = end-start
                #sec_to_hours(end - start) 
                out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)
                out_dec["x_hat"].clamp_(0.,1.)   

                if save_images:
                    output_image = transforms.ToPILImage()(out_dec["x_hat"].squeeze(0))  
                    output_image.save("/scratch/images_k/" + nome_immagine + str(j) + ".png" )

                psnr_im = compute_psnr(x, out_dec["x_hat"])
                ms_ssim_im = compute_msssim(x, out_dec["x_hat"])
                ms_ssim_im = -10*math.log10(1 - ms_ssim_im )
                psnr[j].update(psnr_im)
                mssim[j].update(ms_ssim_im)
                dec_time[j].update(decoded_time)
                #only_dec_time[j].update(out_dec["time"])
            
                size = out_dec['x_hat'].size()
                num_pixels = size[0] * size[2] * size[3]


                data_string_scale = data["strings"][0] # questo è una lista
                bpp_scale = sum(len(s[0]) for s in data_string_scale) * 8.0 / num_pixels #ddddddd
                        
                data_string_hype = data["strings"][1]
                bpp_hype = sum(len(s) for s in data_string_hype) * 8.0 / num_pixels
                bpp = bpp_hype + bpp_scale 


                bpp_loss[j].update(bpp)

                if writing is not None:
                    fls = writing + "/"  +  lev + "_.txt"
                    f=open(fls , "a+")
                    f.write("SEQUENCE "  +   nome_immagine + " BITS " +  str(bpp) + " PSNR " +  str(psnr_im)  + " MSSIM " +  str(ms_ssim_im) + "\n")
                    f.close()  

    #print("enhanced compression results : ",bpp_loss.avg," ",psnr_val.avg," ",mssim_val.avg)
    if writing is not None:
        for j,p in enumerate(pr_list):
            lev =  "level_" + str(j)
            fls = writing +  "/" +  lev + "_.txt"
            f=open(fls , "a+")
            f.write("SEQUENCE "  +   "AVG " + "BITS " +  str(bpp_loss[j].avg) + " YPSNR " +  str(psnr[j].avg)  + " YMSSIM " +  str(mssim[j].avg) + "\n")
            f.close()
    return [bpp_loss[i].avg for i in range(len(bpp_loss))], [psnr[i].avg for i in range(len(psnr))], [dec_time[i].avg for i in range(len(dec_time))] #, [only_dec_time[i].avg for i in range(len(only_dec_time))]

    