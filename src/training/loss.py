import torch.nn as nn 
import torch 
import math
from torch.nn.functional import mse_loss

class ScalableRateDistortionLoss(nn.Module):

    def __init__(self, weight = 255**2, lmbda_list = [0.005, 0.05],  device = "cuda"):
        super().__init__()
        self.scalable_levels = len(lmbda_list)
        self.lmbda = torch.tensor(lmbda_list).to(device) 
        self.weight = weight
        self.device = device

        

    def forward(self,output,target,lmbda = None): 

        batch_size_images, _, H, W = target.size()
        out = {}
        num_pixels = batch_size_images * H * W

        batch_size_recon = output["x_hat"].shape[0] # num_levels,N
        batch_size_recon_tar = target.shape[0]


        if batch_size_recon != 1 and batch_size_recon != batch_size_recon_tar:
            # Copy images to match the batch size of recon_images
            target = target.unsqueeze(0)
            extend_images = target.repeat(batch_size_recon,1,1,1,1) #num_levels, BS,W,H
        else:
            extend_images = target.unsqueeze(0)
        

        if lmbda is  None:
            lmbda = self.lmbda
        else:
            lmbda = torch.tensor([lmbda]).to(self.device)

  
        out["mse_loss"] = mse_loss(extend_images,output["x_hat"],reduction = 'none') # compute the point-wise mse #((scales * (extend_images - output["x_hat"])) ** 2).mean()*self.weight
        out["mse_loss"] = out["mse_loss"].mean(dim=(1,2,3,4)) #dim = num_levels 


        denominator = -math.log(2) * num_pixels  # (batch_size * perce, 1
        likelihoods = output["likelihoods"]
        out["bpp_hype"] =  (torch.log(likelihoods["z"]).sum())/denominator

        if "z_prog" in list(out.keys()):
            out["bpp_hype"] = out["bpp_hype"] +  torch.log(likelihoods["z_prog"]).sum()/denominator


        if "y_prog" in list(likelihoods.keys()):
            out["bpp_base"] = (torch.log(likelihoods["y"]).sum())/denominator
            out["bpp_scalable"] = (torch.log(likelihoods["y_prog"]).sum()).sum()/denominator 
            out["bpp_loss"] = out["bpp_scalable"] +  out["bpp_base"] + batch_size_recon*(out["bpp_hype"])
        else: 
            out["bpp_base"] = (torch.log(likelihoods["y"].squeeze(0)).sum())/denominator
            out["bpp_scalable"] = ((torch.log(likelihoods["y"]).sum()).sum()/denominator)*0.0
            out["bpp_loss"] = out["bpp_scalable"] + out["bpp_base"] + batch_size_recon*(out["bpp_hype"])
        out["loss"] = out["bpp_loss"] + self.weight*(lmbda*out["mse_loss"]).mean() 
        return out




class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()


    def forward(self, output, target, lmbda = 1e-2):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))for likelihoods in output["likelihoods"].values())
        
        
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out
    


class DistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        #self.lmbda = 1e-2


    def forward(self, output, target, lmbda = 1e-2):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))for likelihoods in output["likelihoods"].values())
        
        
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = lmbda*255 ** 2 * out["mse_loss"]

        return out