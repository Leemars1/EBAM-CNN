import torch
import torch.nn as nn
import math
from netCDF4 import Dataset
from tqdm.notebook import tqdm as tqdm
import torch.utils.checkpoint
from torch.optim.optimizer import Optimizer, required
import torch
from einops import rearrange
from torch import nn

# RAdam optimizer implementation (Rectified Adam)
class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=False):
        # Initialize parameters and validate inputs
        # RAdam uses adaptive learning rates with rectified moment estimations
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):
        # Perform a single optimization step (update weights based on gradients)
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # Using average and max pooling to capture spatial features
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward pass to compute channel-wise attention
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# Self-attention mechanism to capture long-range dependencies in spatial dimensions
class Attention(nn.Module):
    def __init__(self, dim, num_heads = 4, bias = True):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # Forward pass to compute the attention map and apply it to the input tensor
        b, c, h, w = x.shape


        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # [B, head, C/head, HW] * [B, head, HW, C/head] * [head, 1, 1] ==> [B, head, C/head, C/head]
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # [B, head, C/head, C/head] * [B, head, C/head, HW] ==> [B, head, C/head, HW]
        out = (attn @ v)

        # [B, head, C/head, HW] ==> [B, head, C/head, H, W]
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

# Spatial attention mechanism to focus on important spatial locations
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Convolutional layer to compute spatial attention based on average and max pooling
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward pass to compute spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# EBAM (Enhanced Block Attention Module) that combines channel and spatial attention mechanisms
class EBAM(nn.Module):
    def __init__(self, dim, num_heads = 4, bias = True, kernel_size=7):
        super(EBAM, self).__init__()
        self.ca = Attention(dim, num_heads, bias)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

# CNN model with embedded attention mechanisms
class EBAM_CNN(nn.Module):
    def __init__(self):
        super(EBAM_CNN, self).__init__()
        # Define the number of channels at each layer
        self.channels_num_1 = 8
        self.channels_num_2 = 32
        self.channels_num_3 = 64
        self.channels_num_4 = 128
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ELU()
        # Define EBAM blocks for each feature map size
        self.ebam_1 = EBAM(self.channels_num_1)
        self.ebam_2 = EBAM(self.channels_num_2)
        self.ebam_3 = EBAM(self.channels_num_3)
        self.ebam_4= EBAM(self.channels_num_4)
        
        # Define convolutional layers to process the input
        self.layer1 = nn.Sequential(
            nn.Conv2d(8, 32, 3, stride = 1, padding=0),
            nn.BatchNorm2d(32),
            #nn.Dropout(0.1)         
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride = 1,padding=0),
            nn.BatchNorm2d(64),
            #nn.Dropout(0.1)  
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64,128, 3,stride = 1,padding=0),
            nn.BatchNorm2d(128),
            #nn.Dropout(0.1)
             
        )

        # Fully connected layers for final prediction
        self.fc1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.Dropout(0.1),  
            nn.ELU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.Dropout(0.1), 
            #nn.LeakyReLU()
            nn.ELU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(32,1)
#             nn.Dropout(0.5),
#             nn.ELU()     
        )

    def forward(self, x):
        # Forward pass through the network with attention applied
        # print(x)
        out = self.ebam_1(x)
        out = self.layer1(out)
        out = self.relu(out)
        out = self.ebam_2(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.ebam_3(out)
        out = self.layer3(out)
        out = self.relu(out)
        out = self.ebam_4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        # out = self.fc4(out)
        return out















