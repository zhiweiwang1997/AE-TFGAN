# -*- coding: utf-8 -*-
"""

@author: Zhiwei Wang, Southeast University, E-mail: zhiwei_wang@seu.edu.cn

"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import scipy.linalg as linalg
from utils import TorchSignalToFrames


# Discriminator loss
def loss_Dis(discriminator, GANlossType, label_batch, est_batch, real_out, fake_out, BatchSize, device): 
    
    if GANlossType == 'GAN':
        
        loss = nn.BCEWithLogitsLoss()(real_out, Variable(torch.ones(BatchSize, 1).cuda())) + \
                    nn.BCEWithLogitsLoss()(fake_out, Variable(torch.zeros(BatchSize, 1).cuda()))
        
    elif GANlossType == 'WGAN': 
        
        loss = -real_out.mean() + fake_out.mean()
        
    elif GANlossType == 'WGAN-GP': 
                
        lambda_gp = 10 # Loss weight for gradient penalty
        loss = -real_out.mean() + fake_out.mean() + lambda_gp*compute_gradient_penalty(discriminator, label_batch, est_batch, device)
        
    elif GANlossType == 'Hinge':
        
        loss = nn.ReLU()(1.0 - real_out).mean() + nn.ReLU()(1.0 + fake_out).mean()
        
        
    elif GANlossType == 'RALS' or GANlossType == 'RALS+TF':
        
        real_logit = real_out - torch.mean(fake_out)
        fake_logit = fake_out- torch.mean(real_out)
        loss = (torch.mean((real_logit - 1.) ** 2) + torch.mean((fake_logit + 1.) ** 2))/2 
        
        
    return loss

# Generator loss
def loss_Gen(GANlossType, label_vec, est_vec, real_out, fake_out, BatchSize, Mae_Mse_Type, alpha, beta, device):
    
    if GANlossType == 'GAN':
        
        loss = nn.BCEWithLogitsLoss()(fake_out, Variable(torch.ones(BatchSize, 1).cuda()))
        
    elif GANlossType == 'WGAN' or GANlossType == 'WGAN-GP' or GANlossType == 'Hinge':
        
        loss = - fake_out.mean()
        
    elif GANlossType == 'RALS':
        
        real_logit = real_out - torch.mean(fake_out)
        fake_logit = fake_out- torch.mean(real_out)
        loss = (torch.mean((real_logit + 1.) ** 2) + torch.mean((fake_logit - 1.) ** 2)) / 2
            
    elif GANlossType == 'RALS+TF':
        
        real_logit = real_out - torch.mean(fake_out)
        fake_logit = fake_out- torch.mean(real_out)
        g_loss = (torch.mean((real_logit + 1.) ** 2) + torch.mean((fake_logit - 1.) ** 2)) / 2
        
        # T loss
        loss_t = t_loss(Mae_Mse_Type, est_vec, label_vec)
        
        # F loss
        frame_size = 1024
        frame_shift = frame_size // 4
        stftf_loss = stftm_loss(frame_size, frame_shift, 'mag', Mae_Mse_Type, device)
        loss_f = stftf_loss(est_vec, label_vec)
        
        # final loss
        loss = g_loss + alpha*loss_t + beta*loss_f 
        
    return loss



# Validation loss 
def loss_Val(label_vec, est_vec, Mae_Mse_Type, alpha, beta, device):
 
    # TF loss
    loss_t = t_loss(Mae_Mse_Type, est_vec, label_vec, 1)
    frame_size = 1024
    frame_shift = frame_size // 4
    stftf_loss = stftm_loss(frame_size, frame_shift, 'mag', Mae_Mse_Type, device)
    loss_f = stftf_loss(est_vec, label_vec)
    loss = alpha*loss_t + beta*loss_f 

    return loss    



class stftm_loss(object):
    def __init__(self, frame_size=512, frame_shift=256, cri_type='mag', loss_type='mae', device='cpu:0'):
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.cri_type = cri_type
        self.loss_type = loss_type
        self.loss_type = loss_type
        self.device = device
        self.frame = TorchSignalToFrames(frame_size=self.frame_size,
                                         frame_shift=self.frame_shift)
        D = linalg.dft(frame_size)
        W = np.hamming(self.frame_size)
        DR = np.real(D)
        DI = np.imag(D)
        self.DR = torch.from_numpy(DR).float().to(self.device)
        self.DR = self.DR.contiguous().transpose(0, 1)
        self.DI = torch.from_numpy(DI).float().to(self.device)
        self.DI = self.DI.contiguous().transpose(0, 1)
        self.W = torch.from_numpy(W).float().to(self.device)

    def __call__(self, outputs, labels):
        outputs = self.frame(outputs)
        labels = self.frame(labels)
        if self.cri_type == 'mag':
            outputs = self.get_stftm(outputs)
            labels = self.get_stftm(labels)
            if self.loss_type == 'mse':
                loss = ((outputs - labels) ** 2.0).mean()
            elif self.loss_type == 'mae':
                loss = (torch.abs(outputs - labels)).mean()
            return loss
        elif self.cri_type == 'ri':
            outputs = self.get_ri(outputs)
            labels = self.get_ri(labels)
            if self.loss_type == 'mse':
                loss = ((outputs - labels) ** 2.0).mean()
            elif self.loss_type == 'mae':
                loss = (torch.abs(outputs - labels)).mean()
            return loss

    def get_stftm(self, frames):
        frames = frames * self.W
        stft_R = torch.matmul(frames, self.DR)
        stft_I = torch.matmul(frames, self.DI)
        stftm = torch.sqrt(torch.square(stft_R) + torch.square(stft_I) + 1e-12)
        return stftm

    def get_ri(self, frames):
        frames = frames * self.W
        stft_R = torch.matmul(frames, self.DR)
        stft_I = torch.matmul(frames, self.DI)
        return torch.stack((stft_R, stft_I), dim=-1)
    
    
def t_loss(loss_type, outputs, labels):
    if loss_type == 'mae':
        loss = (torch.abs(outputs - labels)).mean()
    elif loss_type == 'mse':
        loss = (torch.square(outputs - labels)).mean() 
    return loss


def snr_loss(outputs, labels):
    """
    :param outputs: (1, L)
    :param labels: (1, L)
    :return:
    """
    norm_nomin = torch.sum(labels**2.0, dim=-1, keepdim=True)
    resi = outputs - labels
    norm_denomin = torch.sum(resi**2.0, dim=-1, keepdim=True)
    return (-10*torch.log10(norm_nomin / (norm_denomin + 1e-10) + 1e-10)).mean()

def MRE_loss(outputs, labels):
    resi = outputs - labels
    numerator = torch.sum(resi**2.0)
    denominator = torch.sum(labels**2.0)
    loss = 100 * torch.sqrt(numerator) / torch.sqrt(denominator)
    return loss


def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    if real_samples.dim() == 2:
        real_samples = real_samples.unsqueeze(dim=1)
    if fake_samples.dim() == 2:
        fake_samples = fake_samples.unsqueeze(dim=1)   
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1))).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    interpolates = interpolates.to(device)
    d_interpolates = discriminator(interpolates)
    fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(device)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1).to(device)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


 