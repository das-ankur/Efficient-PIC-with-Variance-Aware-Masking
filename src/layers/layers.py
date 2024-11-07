import torch
import torch.nn as nn
from .win_attention import WinBasedAttention



def deconv(in_channels, out_channels, kernel_size=5, stride=2):     # SN -1 + k - 2p
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)



class ResidualUnit(nn.Module):

    def __init__(self,N):
        super().__init__()
        self.conv = nn.Sequential(
                conv1x1(N, N // 2),
                nn.GELU(),
                conv3x3(N // 2, N // 2),
                nn.GELU(),
                conv1x1(N // 2, N),
            )
        self.relu = nn.GELU()

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out += identity
        out = self.relu(out)
        return out

class Win_noShift_Attention(nn.Module):
    """Window-based self-attention module."""

    def __init__(self, dim, num_heads=8, window_size=8, shift_size=0):
        super().__init__()
        N = dim


        self.conv_a = nn.Sequential(ResidualUnit(N), ResidualUnit(N), ResidualUnit(N))

        self.conv_b = nn.Sequential(
            WinBasedAttention(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=shift_size),
            ResidualUnit(N),
            ResidualUnit(N),
            ResidualUnit(N),
            conv1x1(N, N),
        )

    def forward(self, x):
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out
    

def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def subpel_conv3x3(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )
