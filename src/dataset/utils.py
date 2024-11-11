import multiprocessing
import os
from pathlib import Path
from torch.autograd import Variable

from PIL import Image
from torch.utils.data import Dataset
import torch


class ImageFolder(Dataset):


    def __init__(self, root, num_images = 24000, transform=None, split="train", names = False):
        splitdir = Path(root) / split / "data"
        self.names = names
        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples =[]# [f for f in splitdir.iterdir() if f.is_file()]

        num_images = num_images
            
        print("entro qui per il dataset")
        for i,f in enumerate(splitdir.iterdir()):
            if i%50000==0:
                print(i)
            if i <= num_images: 
                if f.is_file() and i < num_images:
                    self.samples.append(f)
            else:
                break
        print("lunghezza: ",len(self.samples))
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB") #dd
        #st = str(self.samples[index])
        #nome = st.split("/")[-1].split(".")[0]
        
        return self.transform(img)



    def __len__(self):
        return len(self.samples)
    



class TestKodakDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform
        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.image_path = [os.path.join(self.data_dir,f) for f in os.listdir(self.data_dir)]

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        #transform = transforms.Compose([transforms.CenterCrop(256), transforms.ToTensor()])
        
        return self.transform(image), image_ori

    def __len__(self):
        return len(self.image_path)
    


