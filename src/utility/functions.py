
from os.path import join   
from PIL import Image
from torchvision import transforms
from pytorch_msssim import ms_ssim
import math 
import torch

from collections import OrderedDict


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

from torch import Tensor



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