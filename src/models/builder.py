import torch.nn as nn 
from layers import conv, GDN, Win_noShift_Attention, conv3x3, subpel_conv3x3, deconv

def define_decoder(multiple_decoder,N,M,dimensions_M):
        if multiple_decoder:
    
            g_s = nn.ModuleList(
                        nn.Sequential(
                        Win_noShift_Attention(dim= dimensions_M[0], num_heads=8, window_size=4, shift_size=2),
                        deconv(dimensions_M[0], N, kernel_size=5, stride=2),
                        GDN(N, inverse=True),
                        deconv(N, N, kernel_size=5, stride=2),
                        GDN(N, inverse=True),
                        Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
                        deconv(N, N, kernel_size=5, stride=2),
                        GDN(N, inverse=True),
                        deconv(N, 3, kernel_size=5, stride=2),
                ) for _ in range(2) # per adesso solo due, poi vediamo
            )
        else:
            g_s = nn.Sequential(
                Win_noShift_Attention(dim=dimensions_M[0], num_heads=8, window_size=4, shift_size=2),
                deconv(dimensions_M[0], N, kernel_size=5, stride=2),
                GDN(N, inverse=True),
                deconv(N, N, kernel_size=5, stride=2),
                GDN(N, inverse=True),
                Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
                deconv(N, N, kernel_size=5, stride=2),
                GDN(N, inverse=True),
                deconv(N, 3, kernel_size=5, stride=2),
                )
        return g_s






def define_encoder(multiple_encoder, N,M, dimensions_M):

        if multiple_encoder:
            g_a = nn.ModuleList(
                    nn.Sequential(
                    conv(3, N, kernel_size=5, stride=2), # halve 128
                    GDN(N),
                    conv(N, N, kernel_size=5, stride=2), # halve 64
                    GDN(N),
                    Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4), # 
                    conv(N, N, kernel_size=5, stride=2), #32 
                    GDN(N),
                    conv(N, dimensions_M[0], kernel_size=5, stride=2), # 16
                    Win_noShift_Attention(dimensions_M[0], num_heads=8, window_size=4, shift_size=2),
                ) for _ in range(2)
            )
        else:
            g_a = nn.Sequential(
                conv(3, N, kernel_size=5, stride=2), # halve 128
                GDN(N),
                conv(N, N, kernel_size=5, stride=2), # halve 64
                GDN(N),
                Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4), # 
                conv(N, N, kernel_size=5, stride=2), #32 
                GDN(N),
                conv(N, M, kernel_size=5, stride=2), # 16
                Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
            )
        return g_a



def define_hyperprior(multiple_hyperprior,M,N,dimensions_M):
    h_a = nn.Sequential(
            conv3x3(M, 320),
            nn.GELU(),
            conv3x3(320, 288),
            nn.GELU(),
            conv3x3(288, 256, stride=2),
            nn.GELU(),
            conv3x3(256, 224),
            nn.GELU(),
            conv3x3(224, N, stride=2),
        )


    if multiple_hyperprior:

        h_mean_s = nn.ModuleList(
                nn.Sequential(
                conv3x3(N, 192),
                nn.GELU(),
                subpel_conv3x3(192, 224, 2),
                nn.GELU(),
                conv3x3(224, 256),
                nn.GELU(),
                subpel_conv3x3(256, 288, 2),
                nn.GELU(),
                conv3x3(288, dimensions_M[0]),
        ) for i in range(2))

        h_scale_s = nn.ModuleList(
                    nn.Sequential(
                    conv3x3(N, 192),
                    nn.GELU(),
                    subpel_conv3x3(192, 224, 2),
                    nn.GELU(),
                    conv3x3(224, 256),
                    nn.GELU(),
                    subpel_conv3x3(256, 288, 2),
                    nn.GELU(),
                    conv3x3(288, dimensions_M[0]),
                ) for i in range(2))
    else:
        h_mean_s = nn.Sequential(
                conv3x3(N, N),
                nn.GELU(),
                subpel_conv3x3(N, 224, 2),
                nn.GELU(),
                conv3x3(224, 256),
                nn.GELU(),
                subpel_conv3x3(256, 288, 2),
                nn.GELU(),
                conv3x3(288, M),
            )

        h_scale_s = nn.Sequential(
                conv3x3(192, 192),
                nn.GELU(),
                subpel_conv3x3(192, 224, 2),
                nn.GELU(),
                conv3x3(224, 256),
                nn.GELU(),
                subpel_conv3x3(256, 288, 2),
                nn.GELU(),
                conv3x3(288, M),
            )
    return h_a, h_mean_s, h_scale_s