import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class TorchSignalToFrames(object):
    def __init__(self, frame_size=512, frame_shift=256):
        super(TorchSignalToFrames, self).__init__()
        self.frame_size = frame_size
        self.frame_shift = frame_shift
    def __call__(self, in_sig):
        """
        if>
        :param in_sig: (1, L), here L is the sequence of your observed data
        :return: (B, L'), here B actually denotes the number of frames, and L' is the chunked sequence length
        esle>          
        :param in_sig: (3, L), here L is the sequence of your observed data
        :return: (B, 3, L'), here B actually denotes the number of frames, and L' is the chunked sequence length
        """  
        if in_sig.shape[0] == 1: 
          sig_len = in_sig.shape[-1]
          nframes = (sig_len // self.frame_shift)
          a = torch.zeros((nframes, self.frame_size), device=in_sig.device)
          start = 0
          end = start + self.frame_size
          k = 0
          for i in range(nframes):
              if end < sig_len:
                  a[i, :] = in_sig[..., start:end]
                  k += 1
              else:
                  tail_size = sig_len - start
                  a[i, :tail_size] = in_sig[..., start:]

              start = start + self.frame_shift
              end = start + self.frame_size
        else:       
          in_sig = in_sig.unsqueeze(dim=0) # (3, L) -> (1, 3, L)
          sig_len = in_sig.shape[-1]
          nframes = (sig_len // self.frame_shift)   
          a = torch.zeros((nframes, in_sig.shape[-2], self.frame_size), device=in_sig.device)
          for j in range(in_sig.shape[-2]):  
              start = 0
              end = start + self.frame_size
              k = 0
              for i in range(nframes):
                if end < sig_len:
                  a[i, j, :] = in_sig[..., j, start:end]
                  k += 1
                else:
                  tail_size = sig_len - start
                  a[i, j, :tail_size] = in_sig[..., j, start:]
                start = start + self.frame_shift
                end = start + self.frame_size
        return a

      
class TorchOLA(nn.Module):
    r"""Overlap and add on gpu using torch tensor"""
    # Expects signal at last dimension
    def __init__(self, frame_shift=1024):
        super(TorchOLA, self).__init__()
        self.frame_shift = frame_shift

    def forward(self, inputs):
        """
        :param inputs: (B, L')
        :return: (1, L)
        """
        inputs = inputs.squeeze(dim=1)
        nframes = inputs.shape[-2]
        frame_size = inputs.shape[-1]
        frame_step = self.frame_shift
        sig_length = (nframes - 1) * frame_step + frame_size
        sig = torch.zeros((1, sig_length), dtype=inputs.dtype, device=inputs.device,
                         requires_grad=False)
        ones = torch.zeros_like(sig)
        start = 0
        end = start + frame_size
        for i in range(nframes):
           sig[..., start:end] += inputs[..., i, :]
           ones[..., start:end] += 1.
           start = start + frame_step
           end = start + frame_size
        return sig / ones
# ----------------------------------------------------------------------------

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_spectrum(x, n_fft=2048):
    S = librosa.stft(x, n_fft=n_fft)
    # p = np.angle(S)
    S = np.log1p(np.abs(S))
    return S

def save_spectrum(S, lim=800, outfile='spectrogram.png'):
    plt.imshow(S.T, aspect=10)
    # plt.xlim([0,lim])
    plt.tight_layout()
    # plt.savefig(outfile)
  
  
def seed_torch(seed):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
 
