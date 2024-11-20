
from os.path import join   
from PIL import Image
from torchvision import transforms
from pytorch_msssim import ms_ssim
import math 
import torch
from collections import OrderedDict
import torch.optim as optim
import wandb



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
    if args.training_type == "first_strain":
        return optimizer, aux_optimizer
    else:
        return optimizer, None


def read_image(filepath,):
    #assert filepath.is_file()
    img = Image.open(filepath)   
    img = img.convert("RGB")
    return transforms.ToTensor()(img) 

def replace_keys(checkpoint, multiple_encoder):
    # Creiamo un nuovo OrderedDict con le chiavi modificate all'interno di un ciclo for
    nuovo_ordered_dict = OrderedDict()
    for chiave, valore in checkpoint.items():
        if multiple_encoder:
            if "g_a_enh." in chiave: 
                
                nuova_chiave = chiave.replace("g_a_enh.", "g_a.1.")
                nuovo_ordered_dict[nuova_chiave] = valore
            elif "g_a." in chiave and "g_a.0.1.beta" not in list(checkpoint.keys()): 
                nuova_chiave = chiave.replace("g_a.", "g_a.0.")
                nuovo_ordered_dict[nuova_chiave] = valore  
            else: 
                nuovo_ordered_dict[chiave] = valore   
        else:
            nuovo_ordered_dict[chiave] = valore  
    return nuovo_ordered_dict 



class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



import torch

def initialize_model_from_pretrained( checkpoint,
                                     args,
                                     checkpoint_enh = None):

    sotto_ordered_dict = OrderedDict()

    for c in list(checkpoint.keys()):
        if "g_s" in c: 
            if args.multiple_decoder:
                nuova_stringa = "g_s.0." + c[4:]
                sotto_ordered_dict[nuova_stringa] = checkpoint[c]
            else:
                sotto_ordered_dict[c] = checkpoint[c]

        elif "g_a" in c: 
            if args.multiple_encoder:
                nuova_stringa = "g_a.0." + c[4:]
                sotto_ordered_dict[nuova_stringa] = checkpoint[c]
            else:
                sotto_ordered_dict[c] = checkpoint[c]
        elif "cc_" in c or "lrp_" in c or "gaussian_conditional" or "entropy_bottleneck" in c: 
            sotto_ordered_dict[c] = checkpoint[c]
        else:
            continue


    for c in list(sotto_ordered_dict.keys()):
        if "h_scale_s" in c or "h_a" in c  or "h_mean_s" in c:
            sotto_ordered_dict.pop(c)
    if args.multiple_hyperprior:
        for c in list(checkpoint.keys()):
            if "h_mean_s" in c:
                nuova_stringa = "h_mean_s.0." + c[9:]
                sotto_ordered_dict[nuova_stringa] = checkpoint[c]     
            elif "h_scale_s" in c:
                nuova_stringa = "h_scale_s.0." + c[10:]
                sotto_ordered_dict[nuova_stringa] = checkpoint[c]   


    for c in list(sotto_ordered_dict.keys()):
        if "h_a" in c:
            sotto_ordered_dict.pop(c)


    if checkpoint_enh is not None:
        print("prendo anche il secondo modello enhanced")
        for c in list(checkpoint_enh.keys()):
            if "g_s" in c: 
                nuova_stringa = "g_s.1." + c[4:]
                sotto_ordered_dict[nuova_stringa] = checkpoint_enh[c]

            else:
                continue




    return sotto_ordered_dict
def create_savepath(base_path):
    very_best  = join(base_path,"_very_best.pth.tar")
    last = join(base_path,"_last.pth.tar")
    return last, very_best



def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()

def sec_to_hours(seconds):
    a=str(seconds//3600)
    b=str((seconds%3600)//60)
    c=str((seconds%3600)%60)
    d=["{} hours {} mins {} seconds".format(a, b, c)]
    print(d[0])
 


def compute_padding(in_h: int, in_w: int, *, out_h=None, out_w=None, min_div=1):
    """Returns tuples for padding and unpadding.

    Args:
        in_h: Input height.
        in_w: Input width.
        out_h: Output height.
        out_w: Output width.
        min_div: Length that output dimensions should be divisible by.
    """
    if out_h is None:
        out_h = (in_h + min_div - 1) // min_div * min_div
    if out_w is None:
        out_w = (in_w + min_div - 1) // min_div * min_div

    if out_h % min_div != 0 or out_w % min_div != 0:
        raise ValueError(
            f"Padded output height and width are not divisible by min_div={min_div}."
        )

    left = (out_w - in_w) // 2
    right = out_w - in_w - left
    top = (out_h - in_h) // 2
    bottom = out_h - in_h - top

    pad = (left, right, top, bottom)
    unpad = (-left, -right, -top, -bottom)

    return pad, unpad