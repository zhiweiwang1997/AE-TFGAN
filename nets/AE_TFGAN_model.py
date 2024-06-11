# -*- coding: utf-8 -*-
"""

@author: Zhiwei Wang, Southeast University, E-mail: zhiwei_wang@seu.edu.cn

"""

import torch
import torch.nn as nn
from nets.spectral import SpectralNorm
from nets.complexnn import ComplexConv2d
# from spectral import SpectralNorm
# from complexnn import ComplexConv2d

"""
AE-TFGAN:
     - Generator (input features: 3, output features: 1)
     - Discriminator 
"""

class Generator_SN(nn.Module):
    """
    Generator (with SpectralNorm):
         - Encoder:
             - ConvBlock
             - attention_gate
         - Decoder:
             - SubpConvBlock:
                - Subpixel_2D
    """  
    def __init__(self,
                 c_list = [64,64,64,128,128,128,256,256,256],
                 k_h_en = [3,3,3,2,3,3,2,1,1], # kernel height 
                 k_h_de = [1,1,2,3,3,2,3,3,3],
                 k_w = 11,                     # kernel width
                 p_h_en = [1,1,1,0,1,1,0,0,0], # padding height
                 p_h_de = [0,0,1,1,1,1,1,1,1], # padding width
                 ):
        super(Generator_SN, self).__init__()
        self.c_list = c_list
        self.k_h_en = k_h_en
        self.k_h_de = k_h_de
        self.k_w = k_w
        self.p_h_en = p_h_en
        self.p_h_de = p_h_de
        
        # Components
        self.en = Encoder(c_list, k_h_en, k_w, p_h_en)
        self.de = Decoder(c_list, k_h_de, k_w, p_h_de)

    def forward(self, inpt):
        """
        :param inpt: (B,(1),3,L), B: batch size, L: the number of samples
        :return: (B,1,L)
        """
        x, x_list = self.en(inpt)
        x = self.de(x, x_list)

        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, k_h, k_w, pad_h, pad_w, drop_ratio):
        super(ConvBlock, self).__init__()
        """
        # kernel_size = (k_h, k_w)
        # stride = (s_h, s_w)
        # padding = (p_h, p_w)
        """
        self.conv = nn.Sequential(SpectralNorm(nn.Conv2d(in_channel, out_channel, (k_h, k_w), stride=(1,2), padding=(pad_h, pad_w))),
                                   nn.BatchNorm2d(out_channel),
                                   nn.PReLU(out_channel),
                                   nn.Dropout(drop_ratio)
                                   )
    def forward(self, inpt):
        x = inpt
        x = self.conv(x)
        return x

class SubpConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, k_h, k_w, pad_h, pad_w, drop_ratio):
        super(SubpConvBlock, self).__init__()
        """
        # kernel_size = (k_h, k_w)
        # stride = (s_h, s_w)
        # padding = (p_h, p_w)
        """
        self.subp_conv = nn.Sequential(Subpixel_2D(in_channel, out_channel, k_h, k_w, pad_h, pad_w),
                                  nn.BatchNorm2d(out_channel),
                                  nn.PReLU(out_channel),
                                  nn.Dropout(drop_ratio)
                                  )     
    def forward(self, inpt):
        x = inpt
        x = self.subp_conv(x)
        return x
    
class Subpixel_2D(nn.Module):
    def __init__(self, in_channel, out_channel, k_h, k_w, pad_h, pad_w):
        super(Subpixel_2D, self).__init__()
        """
        # kernel_size = (k_h, k_w)
        # stride = (s_h, s_w)
        # padding = (p_h, p_w)
        """
        self.cout = out_channel
        self.conv = nn.Sequential(SpectralNorm(nn.Conv2d(in_channel, out_channel*2, (k_h, k_w), stride=(1,1), padding=(pad_h, pad_w))))

    def forward(self, x):
        b_size, c, h, w = x.shape
        x = self.conv(x)
        b, c, h, w =  x.shape
        x = x.view(b_size, 2, self.cout, h, w)
        x = x.permute(0,2,3,4,1).contiguous().view(b_size, self.cout, h, -1)
        return x    


class attention_gate(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.Wg = nn.Sequential(
            nn.Conv2d(in_c[0], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(in_c[1], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, g, s):
        # Wg = self.Wg(g)
        # Ws = self.Ws(s)
        out = self.relu(self.Wg(g) + self.Ws(s))
        out = self.output(out)
        return out * s


    
class Encoder(nn.Module):
    def __init__(self, c_list, k_h_en, k_w, p_h_en):
        super(Encoder, self).__init__()
        
        cin_list = [1] + c_list[:-1]
        cout_list = c_list
        ConvBlock_list = []
        
        for i in range(len(c_list)):
                        
            if (i+1) % 3 == 0:  # Dropout is employed every three layers
                conv_temp = ConvBlock(cin_list[i], cout_list[i], k_h_en[i], k_w, p_h_en[i], k_w//2, 0.2)
            else:
                conv_temp = ConvBlock(cin_list[i], cout_list[i], k_h_en[i], k_w, p_h_en[i], k_w//2, 0)
            
            ConvBlock_list.append(conv_temp)           
            
        self.ConvBlock_list = nn.ModuleList(ConvBlock_list)

    def forward(self, inpt):
        if inpt.dim() == 3:
            x = inpt.unsqueeze(dim=1)   # (B,3,L) -> (B,1,3,L)
        elif inpt.dim() == 4:
            x = inpt
        else:
            raise SyntaxError('The size of inpt tensor is not valid!')

        x_list = []
        for i in range(len(self.ConvBlock_list)):
            
            x = self.ConvBlock_list[i](x)
            x_list.append(x)        
                       
        return x, x_list



class Decoder(nn.Module):
    def __init__(self, c_list, k_h_de, k_w, p_h_de):
        super(Decoder, self).__init__()
        
        cin_list = c_list[::-1]
        cout_list = cin_list[1:] + [16]
        k_h_de = k_h_de + [3]
        p_h_de = p_h_de + [0]
        SubpConvBlock_list = []
        AttnGateBlock_list = []
       
        for i in range(len(c_list)):
            if i == 0:
                conv_temp = SubpConvBlock(cin_list[i], cout_list[i], k_h_de[i], k_w, p_h_de[i], k_w//2, 0)
            elif i == 2 or i == 5:  # Dropout is employed every three layers
                conv_temp = SubpConvBlock(cin_list[i]*2, cout_list[i], k_h_de[i], k_w, p_h_de[i], k_w//2, 0.2)
            else:
                conv_temp = SubpConvBlock(cin_list[i]*2, cout_list[i], k_h_de[i], k_w, p_h_de[i], k_w//2, 0)
           
            SubpConvBlock_list.append(conv_temp)
            
        for i in range(len(c_list)-1):
            
            attn_temp = attention_gate([ cout_list[i], cout_list[i] ], cout_list[i])
                
            AttnGateBlock_list.append(attn_temp)
                       
        self.SubpConvBlock_list = nn.ModuleList(SubpConvBlock_list)
        self.AttnGateBlock_list = nn.ModuleList(AttnGateBlock_list)
        
        self.last_conv = nn.Sequential(SpectralNorm(nn.Conv2d(cout_list[-1], 1, (k_h_de[-1],k_w), (1,1), (p_h_de[-1], k_w//2))))

    def forward(self, x, x_list):
        
        for i in range(len(self.SubpConvBlock_list)):
            
            if i == 0:                
                x = self.SubpConvBlock_list[i](x)
                x_gate = self.AttnGateBlock_list[i](x_list[-(i+2)], x)
                
            elif i == 8:
                x = self.SubpConvBlock_list[i]( torch.cat((x, x_gate), dim=1) ) 
                
            else:                              
                # x_temp = torch.cat((x, x_gate), dim=1)
                x = self.SubpConvBlock_list[i]( torch.cat((x, x_gate), dim=1) ) 
                x_gate = self.AttnGateBlock_list[i](x_list[-(i+2)], x)                                 
                
        x = self.last_conv(x)
        x = x.squeeze(dim=2)   # (B,1,1,L) -> (B,1,L)
        
        return x

    





class Discriminator_SN(nn.Module):
    """
    Discriminator (with SpectralNorm):
         - timeDiscriminator    time-domain
         - DC_Discriminator     frequency-domain  deep-complex-network
    """  
    def __init__(self, win_size=2048, k=11, fft_len=1024):
        super(Discriminator_SN, self).__init__()
        self.win_size = win_size
        self.k = k            
        self.fft_len = fft_len  
        self.tdisc = timeDiscriminator(win_size, k)
        self.fdisc = DC_Discriminator(fft_len)

    def forward(self, x):
        dt = self.tdisc(x)
        df = self.fdisc(x).view(-1, 1)
        y = dt + df
        return y


    
class timeDiscriminator(nn.Module):
    def __init__(self, win_size=2048, k=11):
        super(timeDiscriminator, self).__init__()
        self.win_size = win_size
        self.k = k
        self.conv1 = nn.Sequential(SpectralNorm(nn.Conv1d(in_channels=1, out_channels=16, kernel_size=k, stride=2, padding=k//2)),
                                   nn.BatchNorm1d(16),
                                   nn.PReLU(16))
        
        self.conv2 = nn.Sequential(SpectralNorm(nn.Conv1d(16, 32, k, 2, k//2)),
                                   nn.BatchNorm1d(32),
                                   nn.PReLU(32))
        
        self.conv3 = nn.Sequential(SpectralNorm(nn.Conv1d(32, 32, k, 2, k//2)),
                                   nn.BatchNorm1d(32),
                                   nn.PReLU(32))
        
        self.conv4 = nn.Sequential(SpectralNorm(nn.Conv1d(32, 64, k, 2, k//2)),
                                   nn.BatchNorm1d(64),
                                   nn.PReLU(64))
        
        self.conv5 = nn.Sequential(SpectralNorm(nn.Conv1d(64, 64, k, 2, k//2)),
                                   nn.BatchNorm1d(64),
                                   nn.PReLU(64))
        
        self.conv6 = nn.Sequential(SpectralNorm(nn.Conv1d(64, 128, k, 2, k//2)),
                                   nn.BatchNorm1d(128),
                                   nn.PReLU(128))

        self.conv7 = nn.Sequential(SpectralNorm(nn.Conv1d(128, 128, k, 2, k//2)),
                                   nn.BatchNorm1d(128),
                                   nn.PReLU(128))
        
        self.fully_connected = SpectralNorm(nn.Linear(in_features=win_size, out_features=1))

    def forward(self, inpt):
        """
        inpt: input batch (signal), size: (B,1,L)
        """
        if inpt.shape[-1] == self.win_size:
            if inpt.dim() == 2:
                x = inpt.unsqueeze(dim=1)
            elif inpt.dim() == 3:
                x = inpt 
                
            x = self.conv1(x)   
            x = self.conv2(x)   
            x = self.conv3(x)   
            x = self.conv4(x)   
            x = self.conv5(x)   
            x = self.conv6(x)   
            x = self.conv7(x)   
            # Flatten
            x = x.view(-1, self.win_size)
            x = self.fully_connected(x)           
        else:
            raise SyntaxError('The size of inpt tensor is not valid!')

        return x    
    

class DC_Discriminator(nn.Module):
    def __init__(self, fft_len=1024):
        super(DC_Discriminator, self).__init__()
        self.win_len = fft_len
        self.hop_len = fft_len // 4  # the distance between neighboring sliding window frames
        self.fft_len = fft_len
        self.win_type = torch.hann_window(self.win_len, device='cuda') #
        

        self.dcconv1 = nn.Sequential(ComplexConv2d(2, 16, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1)),
                                     nn.BatchNorm2d(16),
                                     nn.PReLU())
        self.dcconv2 = nn.Sequential(ComplexConv2d(16, 32, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1)),
                                     nn.BatchNorm2d(32),
                                     nn.PReLU())
        self.dcconv3 = nn.Sequential(ComplexConv2d(32, 64, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1)),
                                     nn.BatchNorm2d(64),
                                     nn.PReLU())
        self.dcconv4 = nn.Sequential(ComplexConv2d(64, 128, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1)),
                                     nn.BatchNorm2d(128),
                                     nn.PReLU())
        self.dcconv5 = nn.Sequential(ComplexConv2d(128, 256, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1)),
                                     nn.BatchNorm2d(256),
                                     nn.PReLU())
        self.dcconv6 = nn.Sequential(ComplexConv2d(256, 256, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1)),
                                     nn.BatchNorm2d(256),
                                     nn.PReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.outc = nn.Sequential(nn.Conv2d(256, 1, 1),
                                  nn.Sigmoid())

    def forward(self, inputs):
        """
        inpt: input batch (signal), size: (B,1,L)
        """      
        # torch STFT func
        specs = torch.stft(inputs.squeeze(), self.fft_len, self.hop_len, self.win_len, self.win_type, onesided=True, return_complex=False)        
        real = specs[..., 0]
        imag = specs[..., 1]
        # mag = torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)
        cspecs = torch.stack([real, imag], 1)

        x = cspecs
        x = self.dcconv1(x)
        x = self.dcconv2(x)
        x = self.dcconv3(x)
        x = self.dcconv4(x)
        x = self.dcconv5(x)
        x = self.dcconv6(x)

        x = self.pool(x)
        x = self.outc(x)
        return x
    
    
     
    
if __name__ == '__main__':
    '''
    input: torch.Size([16, 3, 2048])
    output: torch.Size([16, 1, 2048])
    '''
    generator = Generator_SN().cuda().eval()

    x = torch.rand([16, 3, 2048]).cuda()
    y = generator(x)
    print('{}->{}'.format(x.shape, y.shape))
    
    
    '''
    input: torch.Size([16, 1, 2048])
    output: torch.Size([16, 1])
    '''
    discriminator = Discriminator_SN().cuda().eval()

    x = torch.rand([16, 1, 2048]).cuda()
    y = discriminator(x)
    print('{}->{}'.format(x.shape, y.shape))







