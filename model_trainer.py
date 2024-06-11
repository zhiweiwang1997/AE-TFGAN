# -*- coding: utf-8 -*-
"""

@author: Zhiwei Wang, Southeast University, E-mail: zhiwei_wang@seu.edu.cn

"""

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.utils.data as Data
from nets.AE_TFGAN_model import Generator_SN, Discriminator_SN
from utils import TorchOLA, TorchSignalToFrames, count_parameters, seed_torch
from torch.optim import Adam, lr_scheduler
from loss_criterion import  loss_Dis, loss_Gen, loss_Val
import scipy.linalg as linalg
import time

#%%  
# parameter configurationsa
win_size =  4096     # feature length
shift_ratio = 0.25   # overlap
win_shift = int(shift_ratio*win_size)    
fft_len = 1024       # FFT window len
k = 13               # kernel size 
k_g = k              # kernel size for G
k_d = k              # kernel size for D
c_list = [64,64,64,128,128,128,256,256,256]  # number of channels in the network

GANlossType = 'RALS+TF'  #   ['GAN','WGAN','WGAN-GP','Hinge','RALS','RALS+TF'...]
Mae_Mse_Type = 'mae'     # options: ['mae', 'mse']
lr_g = 1e-4              # learning rate for G
lr_d = 4e-4              # learning rate for D
BatchSize = 16
num_epochs = 150
epoch_save = 120
alpha = 1                # combination coefficients in loss
beta = 50                # combination coefficients in loss
clip_value = 0.01        # lower and upper clip value for D weights
n_critic = 1             # Train the generator every n_critic steps
seed_torch(1)  

#%% 
# check if a CUDA GPU is available and select our device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

# data preparation
trainSet_X = ... # [N,3]
trainSet_Y = ... # [N,1]
valSet_X = ... # [N,3]
valSet_Y = ... # [N,1]


# Standardize features by removing the mean and scaling to unit variance
scalerX = StandardScaler()
scalerX.fit(trainSet_X)
print('Xmean: {}, Xstd: {}'.format(scalerX.mean_, scalerX.scale_))

scalerY = StandardScaler()
scalerY.fit(trainSet_Y)
print('Ymean: {}, Ystd: {}'.format(scalerY.mean_, scalerY.scale_))


trainSet_X = scalerX.transform(trainSet_X)
trainSet_Y = scalerY.transform(trainSet_Y)
valSet_X = scalerX.transform(valSet_X)
valSet_Y = scalerY.transform(valSet_Y)


# define the input and label
inpt = torch.tensor(trainSet_X).t()
label = torch.tensor(trainSet_Y.reshape(1,-1))
inpt_val = torch.tensor(valSet_X).t()
label_val = torch.tensor(valSet_Y.reshape(1,-1))


# enframe the input and label
signal2frame = TorchSignalToFrames(win_size, win_shift)  # chunk the long vector into batchs, with length of each batch being win_size
ola = TorchOLA(win_shift)
frame_inpt, frame_label = signal2frame(inpt), signal2frame(label)
del inpt,label
frame_inpt_val, frame_label_val = signal2frame(inpt_val), signal2frame(label_val)

torch_trainSet = Data.TensorDataset(frame_inpt, frame_label)
train_loader = Data.DataLoader(
                               dataset=torch_trainSet,
                               batch_size=BatchSize,
                               shuffle=False,
                               drop_last=True
                               )

# define the model save path
model_path = './models_save'
if not os.path.exists(model_path):
    os.makedirs(model_path)

# define the results save path
result_path = os.path.join('./results_save', f'Model_{GANlossType}_{Mae_Mse_Type}_{str(win_size)}_{str(alpha)}_{str(beta)}')
if not os.path.exists(result_path):
    os.makedirs(result_path)
    
    
#%%  
# state choices
state = 'train'

if state == 'train':

    # define the model net
    generator = Generator_SN(c_list=c_list, k_w=k_g).to(device)
    discriminator = Discriminator_SN(win_size=win_size, k=k_d, fft_len=fft_len).to(device)
 
    n_para_g = count_parameters(generator)
    n_para_d = count_parameters(discriminator)
    print('N_Para_G: {}, N_Para_D: {}'.format(n_para_g, n_para_d))

    # define the optimizer
    g_optimizer = Adam(generator.parameters(), lr=lr_g, weight_decay=1e-6)
    scheduler_g = lr_scheduler.StepLR(g_optimizer, 5, 0.98)
    d_optimizer = Adam(discriminator.parameters(), lr=lr_d, weight_decay=1e-6)
    scheduler_d = lr_scheduler.StepLR(d_optimizer, 5, 0.98)

    # start training loop
    T1 = time.time()
    
    total_step = len(train_loader)
    log_interval = total_step
    tr_losses_g, tr_losses_d, cv_losses = [], [], []
    start_epoch = 0
    best_epoch = 0
    best_loss = 1e6  # a big value

    for epoch in range(start_epoch, num_epochs):

        # Train the model
        generator.train()
        discriminator.train()

        for batch_idx, (inpt_batch, label_batch) in enumerate(train_loader):

            inpt_batch = inpt_batch.to(device)
            label_batch = label_batch.to(device)
            
            # Train discriminator
            real_out = discriminator(label_batch)                
            est_batch = generator(inpt_batch).detach()          
            fake_out = discriminator(est_batch)                                 
            # Backward and optimizer   
            loss_D = loss_Dis(discriminator, GANlossType, label_batch, est_batch, real_out, fake_out, BatchSize, device)
            d_optimizer.zero_grad()                           
            loss_D.backward()                                  
            d_optimizer.step()   
            tr_losses_d.append(loss_D.item())
            # Clip weights of discriminator
            if GANlossType == 'WGAN':
                for p in discriminator.parameters():
                    p.data.clamp_(-clip_value, clip_value)
                        
            # Train generator
            # Train the generator every n_critic steps
            if batch_idx % n_critic == 0:
                real_out = discriminator(label_batch) 
                est_batch = generator(inpt_batch)                  
                fake_out = discriminator(est_batch)                 
                # Backward and optimizer                        
                label_vec, est_vec = ola(label_batch), ola(est_batch)  # transform to waveform            
                loss_G = loss_Gen(GANlossType, label_vec, est_vec, real_out, fake_out, BatchSize, Mae_Mse_Type, alpha, beta, device)      
                g_optimizer.zero_grad()
                loss_G.backward()                                   
                g_optimizer.step() 
                tr_losses_g.append(loss_G.item())
                            
            # print training stats
            if (batch_idx+1) % log_interval == 0:
                print('Epoch: [{}/{}], Step: [{}/{}] \n Tra_loss_G: {} \n Tra_loss_D: {}'
                    .format(epoch+1, num_epochs, batch_idx+1, total_step, loss_G.item(), loss_D.item()))

        # Val the model
        generator.eval()
        frame_inpt_val = frame_inpt_val.to(device)
        frame_label_val = frame_label_val.to(device)
        frame_output_val = generator(frame_inpt_val)
        label_val_vec, output_val_vec = ola(frame_label_val), ola(frame_output_val)
     
        loss_val = loss_Val(label_val_vec, output_val_vec, Mae_Mse_Type, alpha, beta, device) 
        cv_losses.append(loss_val.item())
        print(' Val_loss: {}'.format(loss_val.item()))
        
        if loss_val.item() <= best_loss and (epoch+1) >= epoch_save:
            best_loss = loss_val.item()
            best_epoch = epoch+1
            torch.save(generator.state_dict(), os.path.join(model_path, 
                          f'Model_{GANlossType}_{Mae_Mse_Type}_{str(win_size)}_{str(alpha)}_{str(beta)}'))
            print('Find better model, save to directory {}'.format(model_path))
       
        scheduler_g.step()
        scheduler_d.step()
        # Clear GPU Memory
        torch.cuda.empty_cache()
        
    print('Epoch: [{}/{}], with best val loss: {}'.format(best_epoch, num_epochs, best_loss))
    
    
    T2 = time.time()
    print('Training time: %s' % (T2-T1))
    
    

#%%
state = 'inference'

if state == 'inference':
    
    model = Generator_SN(c_list=c_list, k_w=k_g).to(device)

    model.load_state_dict(torch.load(os.path.join(model_path, 
                         f'Model_{GANlossType}_{Mae_Mse_Type}_{str(win_size)}_{str(alpha)}_{str(beta)}')))
    model.eval()

    with torch.no_grad():
        signal2frame = TorchSignalToFrames(win_size, win_shift)
        ola = TorchOLA(win_shift)
        frame_inpt_val = signal2frame(inpt_val)
        frame_inpt_val = frame_inpt_val.to(device)
        frame_est_val = model(frame_inpt_val)
        est_val = ola(frame_est_val)[:,:len(valSet_X)].cpu().numpy()
    
    valSet_X_invS = scalerX.inverse_transform(valSet_X).squeeze() 
    valSet_Y_invS = scalerY.inverse_transform(valSet_Y).squeeze()
    est_val_invS = scalerY.inverse_transform(est_val).squeeze()
    

    # Calculate evaluation metrics
    # LWD
    valSet_Y_invS = valSet_Y_invS[0:len(est_val_invS)]  
    toSUM = np.square(np.log10((valSet_Y_invS**2.0) / (est_val_invS**2.0  + 1e-12) + 1e-12))
    val_LWD = np.sqrt(toSUM.mean() + 1e-12)
    print("val LWD: ", val_LWD)

    # LSD
    frame_size = 1024
    frame_shift = 256
    W = np.hamming(frame_size)
    D = linalg.dft(frame_size)
    DR = np.real(D)
    DI = np.imag(D)
    signal2frame = TorchSignalToFrames(frame_size, frame_shift)
    
    valSet_Y_tensor = torch.tensor(valSet_Y_invS.reshape(1,-1))
    frame_label = signal2frame(valSet_Y_tensor)
    frame_label = frame_label * W
    stft_R_label = np.matmul(frame_label, DR)
    stft_I_label = np.matmul(frame_label, DI)
    
    est_val_tensor = torch.tensor(est_val_invS.reshape(1,-1))
    frame_est = signal2frame(est_val_tensor)
    frame_est = frame_est * W
    stft_R_est = np.matmul(frame_est, DR)
    stft_I_est = np.matmul(frame_est, DI)
    
    stftm_label = np.sqrt(np.square(stft_R_label) + np.square(stft_I_label) + 1e-12)
    stftm_est = np.sqrt(np.square(stft_R_est) + np.square(stft_I_est) + 1e-12)
    

    toSUM = np.square(np.log10((stftm_label**2.0) / (stftm_est**2.0  + 1e-12) + 1e-12))
    val_LSD = (np.sqrt(toSUM.mean(axis=1) + 1e-12)).mean().cpu().numpy()
    print("val LSD: ", val_LSD) 
    
     
    