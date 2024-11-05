
import torch.nn as nn
import torch 
class ChannelMask(nn.Module):

    def __init__(self, 
                 mask_policy, 
                 scalable_levels,
                 dim_chunk, 
                 num_levels, 
                 double_dim = False):
        super().__init__()

        self.mask_policy = mask_policy 
        self.scalable_levels = scalable_levels 
        self.quality_list = [i for i in range(self.scalable_levels)]
        self.dim_chunk =dim_chunk 
        self.num_levels = num_levels
        self.double_dim = double_dim
        
        if self.double_dim:
            print("vado qua dove il double dim è raddoppiato!!!")
            self.input_dim = self.dim_chunk*2
        else: 
            self.input_dim = self.dim_chunk

       



    def delta_mask(self, scale,  pr_bar, pr):
        shapes = scale.shape
        bs, ch, w,h = shapes
        assert pr_bar <= pr
        if pr >= 10:
            return torch.ones_like(scale).to(scale.device)
        elif pr == 0:
            return torch.zeros_like(scale).to(scale.device) 
        assert scale is not None 
        pr = 10 if pr > 10 else pr
        pr = pr*0.1
        pr = 1.0 - pr

        pr_bar = 10 if pr_bar > 10 else pr_bar
        pr_bar = pr_bar*0.1
        pr_bar = 1.0 - pr_bar
        res = torch.zeros_like(scale).to(scale.device)   
        for j in range(bs):
            scale_b = scale[j,:,:,:]
            scale_b = scale_b.ravel()
            quantile_bar = torch.quantile(scale_b, pr_bar)
            quantile = torch.quantile(scale_b, pr)
            res_b = quantile_bar >= scale_b >= quantile #if "inverse" not in mask_pol else  scale_b <= quantile
            res_b = res_b.reshape(ch,w,h)
            res_b = res_b.to(scale.device)
            res[j] = res_b             

    def forward(self,
                scale,  
                pr = 0, 
                mask_pol = "point-based-std",
                cust_map = None):

        if cust_map is not None:
            #cust_map = cust_map.unsqueeze(0).to(cust_map.device)
            shapes = cust_map.shape
            bs, ch, w,h = shapes
            if pr >= 10:
                return torch.ones_like(cust_map).to(scale.device)
            elif pr == 0:
                return torch.zeros_like(cust_map).to(scale.device)           
            pr = 10 if pr > 10 else pr
            pr = pr*0.1
            pr_bis = 1.0 - pr
            res = torch.zeros_like(cust_map).to(scale.device)            

            for j in range(bs):
                cust_map_b = cust_map[j,:,:,:]
                cust_map_b = cust_map_b.ravel()
                quantile = torch.quantile(cust_map_b, pr_bis)
                res_b = cust_map_b >= quantile #if "inverse" not in mask_pol else  scale_b <= quantile
                res_b = res_b.reshape(ch,w,h)
                res_b = res_b.to(scale.device)
                res[j] = res_b

            #print("struttura della maschera for: ",pr,": ", torch.unique(res,return_counts = True))
            return res.to(scale.device) 
        
        if mask_pol is None:
            mask_pol = self.mask_policy

        shapes = scale.shape
        bs, ch, w,h = shapes

        if mask_pol is None: 
            return torch.ones_like(scale).to(scale.device)
        if mask_pol == "point-based-std":
            if pr >= 10:
                return torch.ones_like(scale).to(scale.device)
            elif pr == 0:
                return torch.zeros_like(scale).to(scale.device)
            assert scale is not None 
            pr = 10 if pr > 10 else pr
            pr = pr*0.1
            pr_bis = 1.0 - pr
            res = torch.zeros_like(scale).to(scale.device)
            for j in range(bs):
                scale_b = scale[j,:,:,:]
                scale_b = scale_b.ravel()
                quantile = torch.quantile(scale_b, pr_bis)
                res_b = scale_b >= quantile #if "inverse" not in mask_pol else  scale_b <= quantile
                res_b = res_b.reshape(ch,w,h)
                res_b = res_b.to(scale.device)
                res[j] = res_b
            return res.float().reshape(bs,ch,w,h).to(torch.float).to(scale.device)
        elif mask_pol == "two-levels":
            return torch.zeros_like(scale).to(scale.device) if pr == 0 else torch.ones_like(scale).to(scale.device)

        else:
            raise NotImplementedError()