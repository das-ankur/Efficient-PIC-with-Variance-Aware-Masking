from .cnn import WACNN 
from .pic import VarianceMaskingPIC 
from .rem_pic import VarianceMaskingPICREM

models = {"cnn":WACNN,"pic":VarianceMaskingPIC,"rem":VarianceMaskingPICREM}



def get_model(args,device):
    


    if args.model == "cnn":
        net = models[args.model](N = args.N, M = args.M)
    elif args.model == "pic":

        print("stampo il args: ",args)
        net = models[args.model]( N = args.N,
                                M = args.M,
                                multiple_decoder = args.multiple_decoder,
                                multiple_encoder = args.multiple_encoder,
                                multiple_hyperprior = args.multiple_hyperprior,
                                dim_chunk = args.dim_chunk,
                                division_dimension = args.division_dimension,
                                mask_policy = args.mask_policy,
                                support_progressive_slices =args.support_progressive_slices,
                                delta_encode = args.delta_encode,
                                total_mu_rep = args.total_mu_rep,
                                all_scalable = args.all_scalable,
                        )         
    elif args.model == "rem":
        net = models[args.model]( N = args.N,
                                M = args.M,
                                multiple_decoder = args.multiple_decoder,
                                multiple_encoder = args.multiple_encoder,
                                multiple_hyperprior = args.multiple_hyperprior,
                                dim_chunk = args.dim_chunk,
                                division_dimension = args.division_dimension,
                                mask_policy = args.mask_policy,
                                support_progressive_slices =args.support_progressive_slices,
                                delta_encode = args.delta_encode,
                                total_mu_rep = args.total_mu_rep,
                                all_scalable = args.all_scalable,
                                check_levels = args.check_levels,
                                mu_std = args.mu_std,
                                dimension = args.dimension,
                        ) 

    else:
        raise NotImplementedError
    net = net.to(device)
    return net
