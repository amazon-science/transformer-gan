import logging
import sys
from time import strftime, gmtime
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class Discriminator_config(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Signal:
    """Running signal to control training process"""
    def __init__(self, signal_file):
        self.signal_file = signal_file
        self.pre_sig = True
        self.adv_sig = True

        self.update()

    def update(self):
        signal_dict = self.read_signal()
        self.pre_sig = signal_dict['pre_sig']
        self.adv_sig = signal_dict['adv_sig']

    def read_signal(self):
        with open(self.signal_file, 'r') as fin:
            return eval(fin.read())


def create_logger(name, silent=False, to_disk=False, log_file=None):
    """Create a new logger"""
    # setup logger
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    formatter = logging.Formatter(fmt='%(message)s', datefmt='%Y/%m/%d %I:%M:%S')
    if not silent:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        log.addHandler(ch)
    if to_disk:
        log_file = log_file if log_file is not None else strftime("log/log_%m%d_%H%M.txt", gmtime())
        if type(log_file) == list:
            for filename in log_file:
                fh = logging.FileHandler(filename, mode='w')
                fh.setLevel(logging.INFO)
                fh.setFormatter(formatter)
                log.addHandler(fh)
        if type(log_file) == str:
            fh = logging.FileHandler(log_file, mode='w')
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            log.addHandler(fh)
    return log


def get_fixed_temperature(temper, i, N, adapt):
    """A function to set up different temperature control policies"""
    if adapt == 'no':
        temper_var_np = 1.0  # no increase, origin: temper
    elif adapt == 'lin':
        temper_var_np = 1 + i / (N - 1) * (temper - 1)  # linear increase
    elif adapt == 'exp':
        temper_var_np = temper ** (i / N)  # exponential increase
    elif adapt == 'log':
        temper_var_np = 1 + (temper - 1) / np.log(N) * np.log(i + 1)  # logarithm increase
    elif adapt == 'sigmoid':
        temper_var_np = (temper - 1) * 1 / (1 + np.exp((N / 2 - i) * 20 / N)) + 1  # sigmoid increase
    elif adapt == 'quad':
        temper_var_np = (temper - 1) / (N - 1) ** 2 * i ** 2 + 1
    elif adapt == 'sqrt':
        temper_var_np = (temper - 1) / np.sqrt(N - 1) * np.sqrt(i) + 1
    else:
        print("Unknown adapt type!")
        raise ValueError

    return temper_var_np


def get_losses(d_out_real, d_out_fake, loss_type='JS'):
    """Get different adversarial losses according to given loss_type"""
    bce_loss = nn.BCEWithLogitsLoss()

    if loss_type == 'standard':  # the non-satuating GAN loss
        d_loss_real = bce_loss(d_out_real, torch.ones_like(d_out_real))
        d_loss_fake = bce_loss(d_out_fake, torch.zeros_like(d_out_fake))
        d_loss = d_loss_real + d_loss_fake

        g_loss = bce_loss(d_out_fake, torch.ones_like(d_out_fake))

    elif loss_type == 'JS':  # the vanilla GAN loss
        d_loss_real = bce_loss(d_out_real, torch.ones_like(d_out_real))
        d_loss_fake = bce_loss(d_out_fake, torch.zeros_like(d_out_fake))
        d_loss = d_loss_real + d_loss_fake

        g_loss = -d_loss_fake

    elif loss_type == 'KL':  # the GAN loss implicitly minimizing KL-divergence
        d_loss_real = bce_loss(d_out_real, torch.ones_like(d_out_real))
        d_loss_fake = bce_loss(d_out_fake, torch.zeros_like(d_out_fake))
        d_loss = d_loss_real + d_loss_fake

        g_loss = torch.mean(-d_out_fake)

    elif loss_type == 'hinge':  # the hinge loss
        d_loss_real = torch.mean(nn.ReLU(1.0 - d_out_real))
        d_loss_fake = torch.mean(nn.ReLU(1.0 + d_out_fake))
        d_loss = d_loss_real + d_loss_fake

        g_loss = -torch.mean(d_out_fake)

    elif 'wgan' in loss_type:  # 'wgan' or 'wgan-gp'
        d_loss_real = d_out_real.mean()
        d_loss_fake = d_out_fake.mean()
        d_loss = -d_loss_real + d_loss_fake
        g_loss = -d_loss_fake

    elif loss_type == 'tv':  # the total variation distance
        d_loss = torch.mean(nn.Tanh(d_out_fake) - nn.Tanh(d_out_real))
        g_loss = torch.mean(-nn.Tanh(d_out_fake))

    elif 'rsgan' in loss_type:  # 'rsgan' or 'rsgan-gp'
        d_loss = bce_loss(d_out_real - d_out_fake, torch.ones_like(d_out_real))
        g_loss = bce_loss(d_out_fake - d_out_real, torch.ones_like(d_out_fake))

    elif 'ppo' in loss_type:  # 'ppo' or 'ppo-gp'
        with torch.no_grad():
            W = d_out_fake.shape[0] * F.softmax(d_out_fake.data, dim=0)
        # loss_d_clas = (W*soft_logits).mean() - clas_logits.mean() + gp
        d_loss = torch.mean(W * d_out_fake - d_out_real)
        g_loss = -torch.mean(d_out_fake)

    else:
        raise NotImplementedError("Divergence '%s' is not implemented" % loss_type)

    return g_loss, d_loss


def truncated_normal_(tensor, mean=0, std=1):
    """
    Implemented by @ruotianluo
    See https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor
